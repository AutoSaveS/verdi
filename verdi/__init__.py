"""VERDI: proxy-label construction, data processing and a reference
implementation of the three-stage framework described in the manuscript
"From observation to evaluation: Coupling vegetation state and environmental
conditions for near-real-time assessment of urban vegetation resilience
indicators" (Urban Forestry & Urban Greening, under review).

The ``labels`` and ``data`` subpackages implement the operational definitions
and processing rules of Section 3.2 and Appendix A. ``model`` is a reference
implementation written from the architecture description in Section 3.3 and
Appendix G; it is not the code that produced the manuscript's reported numbers.
"""

__version__ = "0.1.0"

#: Study cities, in the order used by the manuscript tables.
CITIES = ("nyc", "paris", "melbourne")
