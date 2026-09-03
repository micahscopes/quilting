# quilting_oklch

Small GPU-safe OKLCH colour vocabulary for Fe renderers.

The ingot provides typed OKLCH, linear and display RGB values; explicit
shorter/longer/increasing/decreasing hue paths; perceptual segment and
three-stop interpolation; fixed-lightness/fixed-hue chroma gamut mapping; an
sRGB display-transfer approximation; display-space coverage mixing; and
opaque RGBA8 packing.

The atlas mesh demo consumes the public API directly. Nothing in this ingot is
specific to tessellation, browser state, or that demo's palette.
