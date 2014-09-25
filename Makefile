all: presentation.pdf

clean:
	rm -f *~ *.toc *.vrb *.out *.nav *.snm *.aux *.log

presentation.pdf: presentation.tex
	pdflatex presentation
	pdflatex presentation
