# Build the AAAI workshop draft.  `make` builds main.pdf, `make check` lists
# problems a reviewer would see, `make clean` removes build products.
#
# TeX Live's binaries are often not on PATH on macOS; override if needed:
#   make TEXBIN=/Library/TeX/texbin
TEXBIN   ?= /Library/TeX/texbin
PDFLATEX ?= $(TEXBIN)/pdflatex
BIBTEX   ?= $(TEXBIN)/bibtex
LATEXOPT  = -interaction=nonstopmode -halt-on-error -file-line-error

MAIN    = main
SOURCES = $(MAIN).tex numbers.tex references.bib $(wildcard sections/*.tex figures/*.tex)

# AAAI requires Times, Helvetica and Courier. A basic TeX Live ships only Times;
# without the other two, build with Computer Modern stand-ins and say so.
PSFONTS := $(shell $(TEXBIN)/kpsewhich phvr7t.tfm pcrr7t.tfm 2>/dev/null | wc -l | tr -d ' ')
ifeq ($(PSFONTS),2)
  TEXINPUT = $(MAIN)
else
  TEXINPUT = '\def\nopsfonts{}\input{$(MAIN)}'
endif
LATEX = $(PDFLATEX) $(LATEXOPT) -jobname=$(MAIN) $(TEXINPUT)

.PHONY: all check clean

all: $(MAIN).pdf

$(MAIN).pdf: $(SOURCES)
	$(LATEX) >/dev/null || (tail -n 30 $(MAIN).log; exit 1)
	$(BIBTEX) $(MAIN) >/dev/null || (cat $(MAIN).blg; exit 1)
	$(LATEX) >/dev/null
	$(LATEX) >/dev/null || (tail -n 30 $(MAIN).log; exit 1)
	@echo "built $(MAIN).pdf"

# Undefined references or citations, overfull boxes, the fallback-layout warning,
# BibTeX warnings, and draft notes left in the source.
check: $(MAIN).pdf
	@grep -nE "undefined|Overfull|not submittable" $(MAIN).log || true
	@grep -n "Warning" $(MAIN).blg || true
	@grep -n '\\note{' $(MAIN).tex sections/*.tex figures/*.tex || true

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).pdf
