# Individual report

`individual_report.pdf` is the public, rebuilt version of the final individual report. The source was taken from the final report archive, and the public copy omits the student number.

Only figures referenced by `individual_report.tex` are stored in `figures/`; LaTeX intermediates are ignored.

Build with a TeX distribution that includes REVTeX 4.2 and BibTeX:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error individual_report.tex
```

Clean generated build files with:

```bash
latexmk -c
```
