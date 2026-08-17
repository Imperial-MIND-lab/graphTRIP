"""
Figure panel modules.

Importing this package registers every target, in the order the figures appear in the
paper. Each module is imported for its @register side effects.

Author: Hanna M. Tolle
Date: 2026-08-10
License: BSD 3-Clause
"""

from figure_making.panels import fig2_performance          # noqa: F401
from figure_making.panels import fig3_atlas                # noqa: F401
from figure_making.panels import fig4_validation           # noqa: F401
from figure_making.panels import fig5_medusa               # noqa: F401
from figure_making.panels import fig6_interpretation       # noqa: F401
from figure_making.panels import supp_dataset_stats        # noqa: F401
from figure_making.panels import supp_ablations            # noqa: F401
from figure_making.panels import supp_z_outcome_predictability  # noqa: F401
from figure_making.panels import supp_bdi                  # noqa: F401
from figure_making.panels import supp_aal                  # noqa: F401
from figure_making.panels import supp_psilodep1            # noqa: F401
from figure_making.panels import supp_medusa               # noqa: F401
from figure_making.panels import supp_interpretability     # noqa: F401
from figure_making.panels import supp_misc                 # noqa: F401
