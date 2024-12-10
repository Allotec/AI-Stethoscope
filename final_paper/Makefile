.PHONY: all clean test gen_references

REPORT_NAME = ESE_534_Report
OUTPUT_FILES = $(REPORT_NAME).pdf $(REPORT_NAME).aux $(REPORT_NAME).log $(REPORT_NAME).out $(REPORT_NAME).bbl  $(REPORT_NAME).blg

all: $(REPORT_NAME).pdf

$(REPORT_NAME).pdf: $(REPORT_NAME).tex $(REPORT_NAME).bib
	@pdflatex $(REPORT_NAME).tex $(REPORT_NAME).pdf 
	@bibtex $(REPORT_NAME) || true
	@pdflatex $(REPORT_NAME).tex $(REPORT_NAME).pdf 
	@pdflatex $(REPORT_NAME).tex $(REPORT_NAME).pdf 

clean:
	@rm $(OUTPUT_FILES)
