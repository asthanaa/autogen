"""Dense validation oracles, excluded from production imports."""

from .sector_oap import ExactSectorOAPEvaluator, solve_exact_sector_oap

__all__ = ["ExactSectorOAPEvaluator", "solve_exact_sector_oap"]
