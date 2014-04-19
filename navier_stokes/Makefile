MESH = lshape
SCALES = 2 1 0.5
MESHES := $(foreach scale, $(SCALES), $(MESH)_$(scale).xml.gz)

meshes: $(MESHES)

$(MESH)_%.msh: $(MESH).geo
	gmsh -2 -clscale $* -o $@ $<

$(MESH)_%.xml: $(MESH)_%.msh
	dolfin-convert $< $@

$(MESH)_%.xml.gz: $(MESH)_%.xml
	gzip $<
