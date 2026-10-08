import pytest

from pedigree_ea import expected_kinship, same_structure
from pedigree_ea import io, synth


def test_pedigree_round_trip(tmp_path):
    ped = synth.avuncular()
    path = tmp_path / "ped.tsv"
    io.write_pedigree(ped, path)
    back = io.read_pedigree(path)
    assert back.ids == ped.ids
    assert all(back[p].parents == ped[p].parents for p in ped)
    assert all(back[p].sex == ped[p].sex for p in ped)
    assert back.observed_ids == ped.observed_ids
    assert same_structure(back, ped)


def test_ibd_round_trip(tmp_path):
    data = synth.observe_ibd(synth.sibs_with_children(), noise_sd=0.02, seed=0)
    path = tmp_path / "ibd.csv"
    io.write_ibd(data, path)
    back = io.read_ibd(path)
    assert dict(back.items()) == dict(data.items())


def test_ibd_csv_with_empty_ibd0(tmp_path):
    path = tmp_path / "ibd.csv"
    path.write_text("id1,id2,ibd0,ibd1,ibd2\nA,B,,0.5,0.25\n")
    assert tuple(io.read_ibd(path)["A", "B"]) == (0.25, 0.5, 0.25)


def test_read_plink_genome(tmp_path):
    path = tmp_path / "plink.genome"
    path.write_text(
        " FID1 IID1 FID2 IID2 RT EZ Z0 Z1 Z2 PI_HAT PHE DST PPC RATIO\n"
        " f1 A f1 B OT 0 0.2600 0.4900 0.2500 0.4950 -1 0.8 1.0 9.0\n")
    assert tuple(io.read_plink_genome(path)["A", "B"]) == (0.26, 0.49, 0.25)


def test_read_king_seg(tmp_path):
    path = tmp_path / "king.seg"
    path.write_text(
        "FID1 ID1 FID2 ID2 MaxIBD1 MaxIBD2 IBD1Seg IBD2Seg PropIBD InfType\n"
        "f1 A f1 B 50.0 30.0 0.5000 0.2500 0.5000 FS\n")
    assert tuple(io.read_king_seg(path)["A", "B"]) == pytest.approx((0.25, 0.5, 0.25))


def test_kinship_round_trip(tmp_path):
    data = expected_kinship(synth.sibs_with_children())
    path = tmp_path / "kin.csv"
    io.write_kinship(data, path)
    back = io.read_kinship(path)
    assert dict(back.items()) == dict(data.items())


KIN0 = """#ID1	ID2	NSNP	HETHET	IBS0	KINSHIP
200	100	11206	0.222649	0.0198108	0.235734
300	100	11780	0.20382	0.021562	0.207725
300	200	11183	0.229545	0.0167218	0.256046
400	100	11786	0.154251	0.038096	0.0996637
400	200	11186	0.188092	0.00858216	0.224144
400	300	12015	0.171785	0.0327091	0.134548
500	100	11208	0.116613	0.0753926	-0.0807562
500	200	11105	0.124629	0.0751013	-0.0630654
500	300	11185	0.119893	0.074743	-0.0717346
500	400	11188	0.111727	0.0777619	-0.0875163
"""


def test_read_king_kinship(tmp_path):
    path = tmp_path / "x.kin0"
    path.write_text(KIN0)
    raw = io.read_king_kinship(path)
    assert raw["100", "200"] == pytest.approx(0.235734)
    assert raw["500", "100"] < 0
    assert io.read_king_kinship(path, clip_negative=True)["500", "100"] == 0
    assert len(raw) == 10 and sorted(raw.ids) == ["100", "200", "300", "400", "500"]


def test_read_king_with_family_ids(tmp_path):
    path = tmp_path / "x.kin0"
    path.write_text("#FID1\tIID1\tFID2\tIID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n"
                    "f\tA\tf\tB\t100\t0.2\t0.0\t0.25\n")
    assert io.read_king_kinship(path)["A", "B"] == 0.25


def test_read_king_ibd(tmp_path):
    path = tmp_path / "x.kin0"
    path.write_text(KIN0)
    data, base = io.read_king_ibd(path)
    assert base == pytest.approx((0.0751013 + 0.0753926) / 2)   # median of 4 unrelated pairs
    full_sib = data["100", "300"]
    assert sum(full_sib) == pytest.approx(1)
    assert data["200", "300"] == pytest.approx((0.22, 0.53, 0.24), abs=0.01)
    assert data["200", "400"].k0 < 0.15 and data["200", "400"].k1 > 0.85   # parent-child-like
    assert data["100", "500"].k0 == pytest.approx(1, abs=0.01)


def test_read_king_ibd_needs_ibs0(tmp_path):
    path = tmp_path / "x.kin0"
    path.write_text("#ID1\tID2\tKINSHIP\nA\tB\t0.25\n")
    with pytest.raises(ValueError, match="IBS0"):
        io.read_king_ibd(path)


def test_read_psam(tmp_path):
    path = tmp_path / "x.psam"
    path.write_text("#IID\tSEX\n100\tNA\n200\t1\n300\t2\n")
    assert io.read_psam(path) == {"100": None, "200": "M", "300": "F"}
    path.write_text("#FID\tIID\tSEX\nf\tA\t2\n")
    assert io.read_psam(path) == {"A": "F"}
