Contains miscellanious halo data for testing. Mimics AHF data run on FIRE-2 sims.

Particular files...
`halo_00000_sparse.dat` :
This file is as if the merger tree file was assembled only from snapshots 599 and 600.
The row 599 is actually just a copy of the 600 row, with the halo ID changed manually.
It should not be thought of as a real row.

`snap599*AHF_halos` :
This file is originally from `m12i_res7000_md`. It's been modified to remove most of the halos in the file.
This forces handling particular tests where the merger tree asks for a halo past what the file has.
