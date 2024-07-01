In all the models I'm using, adding `Gaia_G` photometry to the fit absolutely tanks the convergence of the model. Based on eyeballing, it seems to go like this:
* Fitting on $G$, $G_{BP}$, $G_{RP}$ : performance is awful and uncertainty is very high.
* Fitting on $G$, $G_{BP}$ : fit converges and radius uncertainty and error are in the realm of possibility, but it looks like radius is overestimated.
* Fitting on $G$, $G_{RP}$ : this is actually pretty decent
* Fitting on $G_{BP}$, $G_{RP}$ : also pretty okay
* Fitting on $G$, $G_{BP}$, $G_{RP}$, $u$, $g$, $r$, $i$, $z$ : not so great, but not awful
* Fitting on $G_{BP}$, $G_{RP}$, $u$, $g$, $r$, $i$, $z$ : about the same as above

