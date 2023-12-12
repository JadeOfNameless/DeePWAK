DeePWAK.pdf: build/DeePWAK.bbl
	xelatex -output-directory=build DeePWAK.tex
	mv build/DeePWAK.pdf .

build/DeePWAK.bbl: build/DeePWAK.bcf
	biber build/DeePWAK

build/DeePWAK.bcf: build DeePWAK.tex
	xelatex -output-directory=build DeePWAK.tex

build:
	mkdir build

clean:
	rm -rf build
