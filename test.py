from pricing_engine import PricingEngine

engine = PricingEngine()

result = engine.recommend(
    sku="10133",
    current_price=1.0,
    date="2011-11-15"
)

print(result)
