# SPMAC extended abstract — draft

`spmac_abstract.tex` is the draft extended abstract (target 3–6 pages) for
*Selecciones Matemáticas* (SPMAC). It turns `docs/RESULTS.md` (the NO-GO
outcome of the GEM Go/No-Go) into the fallback publication planned in
`docs/PLAN.md`.

## Build

```bash
cd paper/spmac
pdflatex spmac_abstract.tex && pdflatex spmac_abstract.tex
```

The figures are read directly from `results/figures/` (`\graphicspath`), and
the bibliography is inline (`thebibliography`), so no BibTeX run is needed.

The draft has **not been compiled yet**: the cloud session that wrote it has no
TeX installation. It was checked for balanced braces and environments,
resolved citations, references and figure paths, and it parses with pandoc.
Estimated length: about 1,850 words of body text plus 2 tables, 2 full-width
figures and 15 references, so roughly 5–6 pages. If it exceeds the page
limit, drop Figure 1 first.

## Sources of the numbers

All numbers come from the committed results at commit `ae3edd4`:

- `docs/RESULTS.md`, `results/hard_benchmark.json`: Table 1 and the
  clean-field claims.
- `docs/NOISE_ROBUSTNESS.md`, `results/noise_robustness.md`: Table 2 and the
  degraded-field claims.

Not included:

- The ECT/DECT results on the unmerged branch `cloud-ect-vortex-events`. They
  are not in `docs/RESULTS.md` yet.
- ERA5 results. The pipeline in `docs/REALDATA.md` exists but has not been
  run, so the draft only mentions it as outlook.

## TODO before submission

- [ ] Port to the official SPMAC / Selecciones Matemáticas template and check
      the page limit.
- [ ] Author list, affiliations, corresponding e-mail. Decide whether the
      collaborators (J. Lincovil, E. Cayún) are co-authors.
- [ ] Complete the `multiview` reference (original descriptor, PanAm 2026).
- [ ] Archive a code release (e.g. Zenodo DOI) and cite it in the
      availability statement.
- [ ] Check the MSC 2020 codes (55N31, 76F65, 76M22, 62H30).
- [ ] Optional: a Spanish version of the abstract, if the venue asks for one.
