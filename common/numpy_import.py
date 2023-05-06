
# IF YOU WANT TO BOOST CALCULATIONS BY USING YOUR GRAPHICS CARD, ENABLE THIS
graphics_boost = False
# ==========================================================================


if graphics_boost:
    import cupy as np
else:
    import numpy as np
