"""Scientific comparison contracts: pairing, coverage and causal histories."""
import importlib
import unittest
import tempfile
from pathlib import Path
import json

import numpy as np
import pandas as pd


class ComparisonContracts(unittest.TestCase):
    def implementation(self):
        try:
            return importlib.import_module('experiments.comparison_v3.common')
        except ImportError:
            self.fail('Comparison V3 implementation is not available yet')

    def test_rejects_same_count_different_forecast_keys(self):
        m=self.implementation()
        expected=pd.DataFrame({'sku':['a','b'],'week':['2020-01-06','2020-01-06']})
        actual=expected.copy();actual.loc[1,'sku']='c'
        with self.assertRaisesRegex(ValueError,'keys'):
            m.assert_same_keys(actual,expected)

    def test_paired_interval_uses_seed_totals_and_is_reproducible(self):
        m=self.implementation()
        rows=pd.DataFrame([{'method':method,'seed':seed,'revenue':value}
                           for seed in range(5) for method,value in [('song',100.+seed),('ours',110.+seed)]])
        a=m.paired_interval(rows,'ours','song',draws=1000,seed=99)
        b=m.paired_interval(rows.sample(frac=1,random_state=7),'ours','song',draws=1000,seed=99)
        self.assertEqual(a,b);self.assertEqual(a['pairs'],5)
        self.assertGreater(a['ci_low_pct'],0);self.assertEqual(a['wins'],5)

    def test_paired_interval_rejects_missing_or_duplicate_seed(self):
        m=self.implementation()
        rows=pd.DataFrame({'method':['song','song','ours'],'seed':[1,2,1],'revenue':[10.,10.,11.]})
        with self.assertRaisesRegex(ValueError,'seed'):
            m.paired_interval(rows,'ours','song')
        with self.assertRaisesRegex(ValueError,'duplicate'):
            m.paired_interval(pd.concat([rows,rows]),'ours','song')

    def test_grid_tie_breaks_toward_current_then_lower_price(self):
        m=self.implementation()
        self.assertEqual(m.choose_grid([1.,2.,3.],[6.,3.,2.],2.1),2.)
        self.assertEqual(m.choose_grid([1.,3.],[6.,2.],2.),1.)

    def test_simulated_history_uses_generated_units_not_future_labels(self):
        m=self.implementation()
        q=np.arange(12,dtype=float);p=np.ones(12)
        out_q,out_p=m.roll_history(q,p,generated_qty=77.,chosen_price=2.)
        self.assertEqual(out_q[0],77.);self.assertEqual(out_p[0],2.)
        np.testing.assert_array_equal(out_q[1:],q[:-1]);self.assertEqual(q[0],0.)

    def test_metrics_include_real_zero_sales_without_dividing_by_zero(self):
        m=self.implementation()
        result=m.forecast_metrics(np.array([0.,2.]),np.array([1.,1.]))
        self.assertAlmostEqual(result['rmse'],1.)
        self.assertAlmostEqual(result['wape_pct'],100.)
        self.assertEqual(result['mape_positive_n'],1)
        self.assertAlmostEqual(result['mape_positive_pct'],50.)

    def test_preprocessing_is_fitted_on_training_observations_only(self):
        m=self.implementation()
        frame=pd.DataFrame({'sku':['a']*4,'split':['train','train','validation','test'],
                            'observed_sales':[True]*4,'qty':[1.,3.,100.,999.],'price':[2.,4.,10.,99.]})
        scales=m.train_scalers(frame)
        self.assertEqual(scales['a']['qty_mean'],2.)
        self.assertEqual(scales['a']['price_mean'],3.)
        frame.loc[frame['split'].ne('train'),'qty']=-1000.
        self.assertEqual(m.train_scalers(frame),scales)

    def test_fast_kernel_preserves_original_means_and_noise(self):
        try:
            from experiments.comparison_v3.pricing import FastKernel
        except ImportError:
            self.fail('Full-scope pricing implementation is not available yet')
        from experiments.pricing_simulator import KernelDemandSimulator
        weeks=pd.date_range('2019-01-07',periods=30,freq='7D')
        frame=pd.DataFrame({'sku':['a']*30,'week':weeks,'split':['train']*30,
            'observed_sales':[True]*30,'price':np.tile([1.,1.2,1.4],10),
            'qty':20+np.arange(30)%7,'event_days':np.arange(30)%2,'snap_days':np.arange(30)%4})
        original=KernelDemandSimulator().fit(frame);fast=FastKernel().fit(frame)
        for p in [1.,1.11,1.4]:
            self.assertAlmostEqual(original.predict_mean('a',p,weeks[-1]),fast.predict_mean('a',p,weeks[-1]),places=9)
        self.assertAlmostEqual(original.surfaces['a'].residual_log_sd,fast.surfaces['a'].residual_log_sd,places=9)

    def test_pricing_adapter_matches_serving_observation_and_grid(self):
        from experiments.comparison_v3.pricing_models import ArimaxAdapter
        from final_system.ppo_policy import policy_observation
        from final_system.engine import _price_point
        from types import SimpleNamespace
        state={'params':[.1,-.2,.7,1.],'price_min':1.,'price_max':3.,'price_mean':2.,'price_sd':.5,
               'qty_mean':20.,'qty_sd':5.}
        roster=['x@CA_1','y@TX_1'];adapter=ArimaxAdapter({'x@CA_1':state},roster);adapter.reset('x@CA_1')
        row={'week':pd.Timestamp('2020-01-06'),'qty_lag_1':17.,'price_lag_1':1.6}
        expected=policy_observation(roster,'x@CA_1',17.,1.6,1.65,row['week'].date(),state)
        np.testing.assert_allclose(adapter.observation(row,1.65,None),expected,rtol=1e-6)
        prices=np.unique(np.r_[np.linspace(1,3,41),1.65])
        history=SimpleNamespace(qty=17.,price=1.6)
        revenues=[_price_point(state,history,p)['expected_revenue'] for p in prices]
        expected_price=min(prices[np.max(revenues)-revenues<=1e-8],key=lambda p:(abs(p-1.65),p))
        self.assertEqual(adapter.grid(row,1.65),expected_price)

    def test_full_rollout_is_unchanged_by_future_test_labels(self):
        from experiments.comparison_v3.pricing_models import Series,FastKernel,ArimaxAdapter
        from experiments.comparison_v3.pricing import rollout
        dates=tuple(pd.date_range('2019-01-07',periods=80,freq='7D'))
        qty=np.arange(80,dtype=float)%9+10;prices=np.tile([1.,1.2],40)
        state={'params':[.1,-.2,.7,1.],'price_min':1.,'price_max':1.2,'price_mean':1.1,'price_sd':.1,
               'qty_mean':14.,'qty_sd':3.}
        item=Series('x@CA_1',dates,qty.copy(),prices.copy(),np.zeros(80),np.zeros(80),50,50,65,state,np.arange(12,31))
        panel=pd.DataFrame({'sku':['x@CA_1']*50,'week':dates[:50],'qty':qty[:50],'price':prices[:50],
                            'split':['train']*50,'observed_sales':[True]*50,'event_days':[0]*50,'snap_days':[0]*50})
        simulator=FastKernel().fit(panel)
        first=rollout(item,simulator,'arimax_grid',ArimaxAdapter({item.sku:state},[item.sku]),None,42,horizon=10)
        item.qty[65:]=99999.;item.prices[65:]=99999.
        second=rollout(item,simulator,'arimax_grid',ArimaxAdapter({item.sku:state},[item.sku]),None,42,horizon=10)
        self.assertEqual(first,second)

    def test_song_environment_observation_shape_matches_real_adapter(self):
        from experiments.comparison_v3.pricing_models import Series,FastKernel,SongAdapter
        from experiments.comparison_v3.pricing import PricingEnvironment
        from experiments.song_forecast import SongTransformer
        dates=tuple(pd.date_range('2019-01-07',periods=80,freq='7D'))
        qty=np.arange(80,dtype=float)%9+10;prices=np.tile([1.,1.2],40)
        item=Series('x@CA_1',dates,qty,prices,np.zeros(80),np.zeros(80),50,50,65,{},np.arange(12,31))
        panel=pd.DataFrame({'sku':['x@CA_1']*50,'week':dates[:50],'qty':qty[:50],'price':prices[:50],
                            'split':['train']*50,'observed_sales':[True]*50,'event_days':[0]*50,'snap_days':[0]*50})
        simulator=FastKernel().fit(panel)
        adapter=SongAdapter(SongTransformer().state_dict(),{'x@CA_1':{'qty_mean':14.,'qty_std':3.,'price_mean':1.1,'price_std':.1}})
        env=PricingEnvironment({item.sku:item},simulator,adapter,'song',20.,42)
        observation,_=env.reset(seed=42)
        self.assertEqual(observation.shape,env.observation_space.shape)

    def test_song_pricing_features_predictions_and_gradients_match_forecast_module(self):
        import torch
        from experiments.comparison_v3.pricing_models import SongAdapter
        from experiments.song_forecast import SongTransformer,SongDemandForecaster,feature_arrays
        torch.manual_seed(19)
        model=SongTransformer().eval()
        scalers={'x@CA_1':{'qty_mean':100.,'qty_std':5.,'price_mean':2.,'price_std':.5}}
        row={'sku':'x@CA_1','week':pd.Timestamp('2015-12-28'),'event_days':2.,'snap_days':3.}
        for j in range(1,13):
            row.update({f'qty_lag_{j}':90.+j,f'price_lag_{j}':1.8+.02*j,
                        f'event_days_lag_{j}':float(j%3),f'snap_days_lag_{j}':float(j%5)})
        adapter=SongAdapter(model.state_dict(),scalers,19)
        expected=feature_arrays(pd.DataFrame([dict(row,price=2.1)]),scalers)
        actual=adapter.encode(row,2.1)
        for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
        native=SongDemandForecaster(model,scalers)
        qty,gradient=adapter.prediction_gradient(row,2.1)
        self.assertAlmostEqual(qty,native.predict_q(row,2.1),places=5)
        self.assertAlmostEqual(gradient,native.price_gradient(row,2.1),places=5)

    def test_partial_store_predictions_are_not_complete_on_arimax_scope(self):
        from experiments.comparison_v3.charts import forecast_tables
        from experiments.comparison_v3.common import ROOT,SEEDS,digest
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            output=Path(temp);(output/'data/m5').mkdir(parents=True)
            (output/'data/manifest.json').write_text(json.dumps({'frozen_test_keys':{'uci':{'count':0},'m5':{'count':4}}}))
            for store in ['A','B']:
                (output/'data/m5'/f'{store}.json').write_text('{}')
                frame=pd.DataFrame({'sku':[f'x@{store}']*2,'week':['2020-01-06','2020-01-13'],'qty':[1.,2.],
                     'naive_last':[1.,1.],'mean_4':[1.,1.],'arimax_fallback':[1.,1.],'arimax_available':[True,True]})
                path=output/'forecast/m5/baselines';path.mkdir(parents=True,exist_ok=True)
                frame.to_csv(path/f'{store}_test.csv.gz',index=False)
                if store=='A':
                    for seed in SEEDS:
                        target=output/'forecast/m5'/f'seed_{seed}'/'predictions';target.mkdir(parents=True)
                        file=target/'A_test.csv.gz'
                        frame.assign(prediction=[1.,1.]).to_csv(file,index=False)
                        file.with_suffix('.json').write_text(json.dumps({'file':digest(file),'checkpoint':'selected'}))
                        (target.parent/'selected.json').write_text(json.dumps({'checkpoint_sha256':'selected'}))
            result=forecast_tables(output)
            song=result.loc[result['method'].eq('song_transformer')&result['scope'].eq('arimax_available')]
            self.assertEqual(len(song),5)
            self.assertFalse(song['complete_key_scope'].any())

    def test_forecast_chart_rejects_prediction_from_another_checkpoint(self):
        from experiments.comparison_v3.charts import forecast_tables
        from experiments.comparison_v3.common import ROOT,digest
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            output=Path(temp);(output/'data/uci').mkdir(parents=True)
            (output/'data/uci/UCI.json').write_text('{}')
            (output/'data/manifest.json').write_text(json.dumps({'frozen_test_keys':{'uci':{'count':1},'m5':{'count':0}}}))
            frame=pd.DataFrame({'sku':['x'],'week':['2020-01-06'],'qty':[1.],'naive_last':[1.],'mean_4':[1.]})
            baseline=output/'forecast/uci/baselines';baseline.mkdir(parents=True)
            frame.to_csv(baseline/'UCI_test.csv.gz',index=False)
            predictions=output/'forecast/uci/seed_42/predictions';predictions.mkdir(parents=True)
            file=predictions/'UCI_test.csv.gz';frame.assign(prediction=1.).to_csv(file,index=False)
            file.with_suffix('.json').write_text(json.dumps({'file':digest(file),'checkpoint':'old'}))
            (predictions.parent/'selected.json').write_text(json.dumps({'checkpoint_sha256':'new'}))
            with self.assertRaisesRegex(ValueError,'checkpoint'):
                forecast_tables(output)

    def test_one_seed_pricing_cannot_be_marked_full(self):
        from experiments.comparison_v3 import pricing
        from experiments.comparison_v3.common import SEEDS
        self.assertTrue(callable(getattr(pricing,'is_full_result',None)),'Full pricing completion validation missing')
        rows=pd.DataFrame([{'method':method,'seed':seed,'simulator':sim,'series':3,'decisions':60}
            for method in ['fixed_mean','fixed_upper','arimax_grid','arimax_ppo','song_ppo_feedback']
            for seed in SEEDS for sim in ['gaussian','lag_0.75']])
        self.assertTrue(pricing.is_full_result(rows,3))
        self.assertFalse(pricing.is_full_result(rows.loc[rows['seed'].eq(42)],3))
        rows.loc[0,'decisions']=59
        self.assertFalse(pricing.is_full_result(rows,3))

    def test_decision_verifier_rejects_wrong_sku_despite_equal_totals(self):
        from experiments.comparison_v3 import verify
        self.assertTrue(callable(getattr(verify,'check_decision_frame',None)),'Exact decision coverage verifier missing')
        expected=pd.DataFrame({'sku':['x@CA_1'],'week':pd.to_datetime(['2020-01-06']),
                               'train_price_min':[1.],'train_price_max':[2.]})
        frame=pd.DataFrame([{'sku':'x@CA_1','week':'2020-01-06','method':method,'seed':42,'simulator':'gaussian',
             'price':1.5,'simulated_qty':2.,'revenue':3.,'previous_qty':2.,'previous_price':1.5,'train_price_min':1.,'train_price_max':2.}
             for method in ['fixed_mean','fixed_upper','arimax_grid','arimax_ppo','song_ppo_feedback']])
        verify.check_decision_frame(frame,expected,42,'gaussian')
        frame['sku']='wrong@CA_2'
        with self.assertRaisesRegex(ValueError,'keys'):
            verify.check_decision_frame(frame,expected,42,'gaussian')

    def test_decision_verifier_rejects_wrong_identity_and_self_declared_bounds(self):
        from experiments.comparison_v3 import verify
        self.assertTrue(callable(getattr(verify,'check_decision_frame',None)),'Exact decision coverage verifier missing')
        expected=pd.DataFrame({'sku':['x@CA_1'],'week':pd.to_datetime(['2020-01-06']),
                               'train_price_min':[1.],'train_price_max':[2.]})
        frame=pd.DataFrame([{'sku':'x@CA_1','week':'2020-01-06','method':method,'seed':42,'simulator':'gaussian',
             'price':1.5,'simulated_qty':2.,'revenue':3.,'previous_qty':2.,'previous_price':1.5,'train_price_min':1.,'train_price_max':2.}
             for method in ['fixed_mean','fixed_upper','arimax_grid','arimax_ppo','song_ppo_feedback']])
        wrong=frame.assign(seed=123)
        with self.assertRaisesRegex(ValueError,'identity'):verify.check_decision_frame(wrong,expected,42,'gaussian')
        wrong=frame.assign(train_price_max=9.)
        with self.assertRaisesRegex(ValueError,'bounds'):verify.check_decision_frame(wrong,expected,42,'gaussian')

    def test_final_verifier_rejects_copied_seed_and_invalid_predictions(self):
        from experiments.comparison_v3 import verify
        self.assertTrue(callable(getattr(verify,'check_forecast_identity',None)),'Forecast seed verifier missing')
        record={'identity':{'seed':42,'dataset':'m5','data':{'CA_1':'hash'}},'fit_scope':'train_only'}
        verify.check_forecast_identity(record,{'identity':record['identity']},'m5',42,{'CA_1':'hash'})
        with self.assertRaisesRegex(ValueError,'identity'):
            verify.check_forecast_identity(record,{'identity':record['identity']},'m5',123,{'CA_1':'hash'})
        for values in ([float('nan')],[-1.]):
            with self.assertRaisesRegex(ValueError,'prediction'):
                verify.check_prediction_values(pd.DataFrame({'prediction':values}),['prediction'])

    def test_failed_verification_invalidates_old_success_marker(self):
        from experiments.comparison_v3 import verify
        from experiments.comparison_v3.common import ROOT
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            output=Path(temp);marker=output/'verification.json';marker.write_text('{"complete":true}')
            with self.assertRaises(FileNotFoundError):verify.run(output)
            self.assertFalse(marker.exists())

    def test_pricing_identity_changes_when_arimax_parameters_change(self):
        from experiments.comparison_v3 import pricing
        from experiments.comparison_v3.common import ROOT
        self.assertTrue(callable(getattr(pricing,'dependency_identity',None)),'Pricing dependency hashing missing')
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            directory=Path(temp)/'arimax';directory.mkdir()
            file=directory/'CA_1.json';file.write_text('{"beta":1}')
            first=pricing.dependency_identity(Path(temp))
            file.write_text('{"beta":2}')
            self.assertNotEqual(first,pricing.dependency_identity(Path(temp)))

    def test_single_seed_rebuild_removes_stale_confidence_plot(self):
        from experiments.comparison_v3.charts import pricing_figures
        from experiments.comparison_v3.common import ROOT
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            output=Path(temp);(output/'pricing').mkdir();figures=output/'figures';figures.mkdir()
            rows=pd.DataFrame([{'method':method,'seed':42,'simulator':sim,'revenue':100.}
                for method in ['fixed_mean','fixed_upper','arimax_grid','arimax_ppo','song_ppo_feedback']
                for sim in ['gaussian','lag_0.75']])
            rows.to_csv(output/'pricing/runs.csv',index=False)
            for ext in ('png','svg'):(figures/f'pricing_uplift_ci.{ext}').write_text('stale')
            pricing_figures(output,figures)
            self.assertFalse((figures/'pricing_uplift_ci.png').exists())
            self.assertFalse((figures/'pricing_uplift_ci.svg').exists())


if __name__=='__main__':
    unittest.main()
