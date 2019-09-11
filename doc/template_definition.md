# OpeNPDN: Neural networks for automated synthesis of Power Delivery Networks (PDN)

## Instruction for populating template information in *template_definition.json*

The templates are defined in the *template_definition.json* file. In this file the user defines the number of metal layers in the BEOL stack, the number of metal layers used in the PDN, pitch, width, unit resistance of each metal layer, and via resistances.

The following are a list of variables defined in this file:

- NUM_LAYERS: number of layers in the BEOL stack
- TECH_LAYERS: list of all possible layers in the BEOL stack
- PDN_LAYERS: layers used in the PDN
- NUM_REGIONS_X: number of regions/tessellations in the horizontal direction of the design
- NUM_REGIONS_Y: number of regions/tessellations in the horizontal direction of the design
- CHIP_WIDTH: size of the core in the x-direction
- CHIP_HEIGHT: size of the core in the y-direction
- SIZE_REGION_X: width of the region
- SIZE_REGION_Y: height of the region
- NUM_TEMPLATES: total number of templates defined

In the layers section of this file:
- width: width of the power stripe (m)
- min_width: minimum width of the metal layer defined by DRC (m)
- res: resistance per unit micron of the metal layer (Ohm/um)
- via_res: resistance of the via (Ohm)
- pitch: set-to-set spacing between two stripes of the PDN metal layer
- dir: preferred metal layer direction, H/V
- t_spacing: track spacing in the preferred direction (m)

These values can be populated in this file can be populated with reference to the [PDN.cfg](https://github.com/The-OpenROAD-Project/pdn/blob/master/doc/example_PDN.cfg)

