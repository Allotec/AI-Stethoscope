.PHONY: all clean test gen_references

REPORT_NAME = ESE_534_Report
GARB_DIR = .out_files

all: gen_references

gen_references: $(REPORT_NAME).pdf
	@pdflatex $(REPORT_NAME).tex $(REPORT_NAME).pdf
	@mkdir -p $(GARB_DIR)
	@mv ESE_534_Report.aux  ESE_534_Report.log  ESE_534_Report.out $(GARB_DIR)

$(REPORT_NAME).pdf: $(REPORT_NAME).tex
	@pdflatex $(REPORT_NAME).tex $(REPORT_NAME).pdf

clean:
	@rm $(REPORT_NAME).pdf
	@rm $(GARB_DIR)
