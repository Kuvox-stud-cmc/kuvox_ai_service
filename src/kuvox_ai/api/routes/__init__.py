"""HTTP routers, one per endpoint group."""

from kuvox_ai.api.routes.admin import router as admin_router
from kuvox_ai.api.routes.health import router as health_router
from kuvox_ai.api.routes.planning import router as planning_router
from kuvox_ai.api.routes.retrieval import router as retrieval_router

__all__ = ["admin_router", "health_router", "planning_router", "retrieval_router"]
