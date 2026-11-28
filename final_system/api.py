"""HTTP API for the versioned model bundle."""

from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .bundle import DEFAULT_MODEL_DIR, load_bundle
from .engine import InferenceError, forecast, recommend_price
from .ppo_policy import PpoArtifactError, load_ppo_policy
from .schemas import ForecastRequest, ForecastResponse, PriceRequest, PriceResponse


logger = logging.getLogger(__name__)


def create_app(model_dir=None):
    @asynccontextmanager
    async def lifespan(app):
        directory = model_dir or os.environ.get("FINAL_SYSTEM_MODEL_DIR") or DEFAULT_MODEL_DIR
        app.state.model_bundle = load_bundle(directory)
        try:
            app.state.ppo_policy = load_ppo_policy(app.state.model_bundle)
        except PpoArtifactError as error:
            logger.warning("PPO disabled: %s", error)
            app.state.ppo_policy = None
        yield

    app = FastAPI(title="Dynamic Pricing Model API", version="1.1.0", lifespan=lifespan)

    @app.exception_handler(InferenceError)
    async def inference_error(_request: Request, error: InferenceError):
        detail = {"code": error.code, "message": str(error)}
        return JSONResponse(status_code=error.status, content={"detail": detail})

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError):
        # Raw input can contain a value such as 1e400 that JSON cannot echo back.
        detail = [{key: item[key] for key in ("loc", "msg", "type")} for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.get("/health", summary="Check that the model bundle is loaded")
    def health(request: Request):
        bundle = request.app.state.model_bundle
        return {"status": "ready", "model_version": bundle.manifest["version"]}

    @app.get("/model-info", summary="Read model coverage, configuration and provenance")
    def model_info(request: Request):
        manifest = request.app.state.model_bundle.manifest
        config = manifest["config"]
        policy = request.app.state.ppo_policy
        pricing = {key: value for key, value in config["pricing"].items() if key != "reference_skus"}
        return {"model_version": manifest["version"], "protocol_id": config["protocol_id"],
                "datasets": config["datasets"], "m5_fit": config["m5_fit"], "pricing": pricing,
                "m5_pricing_policies": {"grid": {"available": True},
                                        "ppo_feedback": {"available": policy is not None,
                                                         "version": policy.version if policy else None,
                                                         "scope": policy.scope if policy else []}},
                "coverage": manifest["coverage"], "provenance": manifest["provenance"]}

    @app.post("/forecast", response_model=ForecastResponse, response_model_exclude_unset=True,
              summary="Forecast one week's quantity on UCI or M5")
    def forecast_endpoint(payload: ForecastRequest, request: Request):
        return forecast(request.app.state.model_bundle, payload)

    @app.post("/recommend-price", response_model=PriceResponse,
              summary="Recommend a grid or PPO price within an M5 series' training support")
    def price_endpoint(payload: PriceRequest, request: Request):
        return recommend_price(request.app.state.model_bundle, payload, request.app.state.ppo_policy)

    return app


app = create_app()
