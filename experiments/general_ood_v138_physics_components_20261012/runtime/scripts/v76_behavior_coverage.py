"""Coverage-bank metadata for matched behavior-reference sampling."""
from scripts.v74_density_coverage import DensityCoverageBank


class BehaviorCoverageBank(DensityCoverageBank):
    def metadata(self):
        meta = super().metadata()
        meta["geometry_source"] = "v72 base labels plus matched-seed normal/slippery suffix labels, every second transition"
        return meta
