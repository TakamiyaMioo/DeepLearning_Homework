TEX = main.tex
OUTDIR = build

all: $(OUTDIR)/main.pdf

$(OUTDIR)/main.pdf: $(TEX) Settings.tex images/school_flat.png
	mkdir -p $(OUTDIR)
	xelatex -interaction=nonstopmode -output-directory=$(OUTDIR) $(TEX)
	xelatex -interaction=nonstopmode -output-directory=$(OUTDIR) $(TEX)

clean:
	rm -rf $(OUTDIR)

.PHONY: all clean
