MSH_MESHES := $(foreach scale, $(SCALES), $(MESH)_$(scale).msh)
XML_MESHES := $(foreach scale, $(SCALES), $(MESH)_$(scale).xml.gz)

meshes: $(MSH_MESHES) $(XML_MESHES)

$(MESH)_%.msh: $(MESH).geo
	gmsh -2 -clscale $* -o $@ $<

$(MESH)_%.xml: $(MESH)_%.msh
	dolfin-convert $< $@

$(MESH)_%.xml.gz: $(MESH)_%.xml
	gzip -f $<
