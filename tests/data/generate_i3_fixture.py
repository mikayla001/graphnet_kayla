#! /usr/bin/env python3
"""Generate an i3 file of hand-seeded events for I3 extractor unit tests.

Each event starts from a fully specified primary in ``I3MCTree_preMuonProp``
(type, energy, position, direction chosen to realise a particular topology),
which ``PropagateMuons`` (PROPOSAL) then propagates into a physically valid
post-propagation ``I3MCTree`` + ``MMCTrackList``. Every input is public -- the
GCD is GraphNeT's own committed test GCD and PROPOSAL uses open cross-section
tables -- so the result carries no IceCube-internal data. The committed fixture
is what tests read.

The RNG is seeded so regeneration is reproducible. Run inside the GraphNeT
icetray Docker image (icetray + simprod/PROPOSAL):

    python3 tests/data/generate_i3_fixture.py

Every written frame is validated: the post-prop tree and the MMCTrackList must
be present, non-empty, mutually consistent, and harvestable by MuonGun (the
exact dependency the extractors rely on). Generation aborts loudly otherwise.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from icecube import (
    icetray,
    dataclasses,
    phys_services,
    MuonGun,
    simclasses,
)
from icecube.simprod.segments.PropagateMuons import make_standard_propagators

# --- Configuration ----------------------------------------------------------

# Test i3 data tree relative to this file (<repo>/tests/data/ ->
# <repo>/data/tests/i3)
_TEST_I3_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "tests", "i3")
)

# GCD geometry the propagation runs against; the test's GCD_hull must be built
# from the same file so the hull and the propagation agree. Defaults to
# GraphNeT's committed test GCD; override with $I3_FIXTURE_GCD.
GCD_FILE = os.environ.get(
    "I3_FIXTURE_GCD",
    os.path.join(
        _TEST_I3_DIR,
        "oscNext_genie_level7_v02",
        "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise"
        ".i3.gz",
    ),
)
OUTPUT_FILE = os.path.join(
    _TEST_I3_DIR, "i3_fixture", "i3_fixture_events.i3.zst"
)

# Fixed seed -> deterministic regeneration. The committed fixture is what tests
# read; PROPOSAL output is otherwise stochastic and version-sensitive.
RNG_SEED = 12345

# The icetray image ships PROPOSAL's interpolation tables read-only and its
# stock config forces ``just_use_readonly_path``, so a fresh run cannot build
# them. We rewrite the config to a writable scratch dir outside the repo and let
# PROPOSAL build the tables there. Override with $I3_FIXTURE_PROPOSAL_TABLES.
PROPOSAL_CONFIG_IN = os.path.expandvars(
    "$I3_BUILD/PROPOSAL/resources/config_icesim.json"
)
PROPOSAL_TABLES_DIR = os.environ.get(
    "I3_FIXTURE_PROPOSAL_TABLES",
    os.path.join(tempfile.gettempdir(), "i3_fixture_proposal_tables"),
)
PROPOSAL_CONFIG_OUT = os.path.join(
    PROPOSAL_TABLES_DIR, "config_icesim_writable.json"
)

# Rough instrumented-volume dimensions (metres), used only for the reporting in
# describe_topology(). The real containment test is the extractor's convex hull.
DETECTOR_RADIUS = 500.0
DETECTOR_HALF_Z = 500.0

# Each charged lepton is produced at an interaction vertex by a neutrino created
# this far upstream (m) along the incoming direction, so the tree looks like a
# real neutrino event: an in-ice neutrino primary that yields the lepton.
NU_DISTANCE = 2000.0
C_M_PER_NS = 0.299792458  # speed of light, for the neutrino time-of-flight

# Neutrino flavour that produces each charged-lepton type via a CC interaction.
NEUTRINO_FOR = {
    "MuMinus": "NuMu",
    "MuPlus": "NuMuBar",
    "TauMinus": "NuTau",
    "TauPlus": "NuTauBar",
}

# Geometry/inelasticity for the NC-then-CC chain event: the neutrino first does
# a neutral-current interaction this far upstream of the in-detector vertex,
# losing a fraction of its energy to hadrons and continuing as a softer neutrino
# that then interacts (charged-current) inside the detector.
NC_TO_CC_DISTANCE = 2000.0
Y_NC = 0.25  # fraction of energy to hadrons at the outside NC vertex
Y_CC = 0.30  # fraction of energy to hadrons at the in-detector CC vertex

# Cosmic-ray muon bundle (CORSIKA-style): only the in-ice muons are seeded, all
# sharing the shower direction (a real bundle is collinear) and spread
# transversely around the shower axis.
CORSIKA_BUNDLE_ENERGIES = [1500.0, 900.0, 700.0, 600.0, 500.0]  # GeV
CORSIKA_BUNDLE_RADIUS = 10.0  # transverse spread of the bundle (m)
CORSIKA_PRIMARY_UPSTREAM = 1000.0  # m

# Coincident-background event: on top of a signal neutrino whose products all
# stop short of the array, an atmospheric muon from an upstream pi+ decay does
# cross it, so the only light in the detector comes from the background.
COINCIDENT_BG_MUON_ENERGY = 1.0e4  # GeV, enough to cross the whole array
COINCIDENT_BG_PION_UPSTREAM = 200.0  # m, pi+ decay point ahead of the muon


@dataclass
class EdgeCase:
    """A single seed particle and the topology it is meant to produce."""

    name: str
    particle_type: str  # I3Particle.ParticleType attribute name
    energy: float  # GeV
    position: Tuple[float, float, float]  # interaction vertex (x, y, z) [m]
    direction: Tuple[float, float, float]  # travel vector, need not be unit
    comment: str = ""
    # Pre-propagation I3MCTree builder; defaults (None) to a single-vertex CC
    # event (neutrino -> charged lepton).
    builder: Optional[Callable[["EdgeCase"], "dataclasses.I3MCTree"]] = None
    # First in-ice neutrino sits below the top of the tree; the validator
    # asserts this so get_primaries is forced to recurse.
    top_primary_outside_ice: bool = False
    # Cosmic-ray event, extracted with is_corsika=True; stamped on the frame so
    # a consuming test knows to set that flag.
    is_corsika: bool = False


def _normalise(v: Tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Direction vector must be non-zero")
    return arr / norm


def _make_particle(
    type_name: str,
    energy: float,
    pos: np.ndarray,
    direction: np.ndarray,
    time: float = 0.0,
    length: Optional[float] = None,
    location_type: str = "InIce",
) -> "dataclasses.I3Particle":
    """Construct an I3Particle with the given kinematics.

    A non-"InIce" ``location_type`` keeps a primary from being picked as the
    in-ice neutrino, forcing the extractor's recursive search down the tree.
    """
    p = dataclasses.I3Particle()
    p.type = getattr(dataclasses.I3Particle.ParticleType, type_name)
    p.energy = energy
    p.pos = dataclasses.I3Position(*pos)
    p.dir = dataclasses.I3Direction(*direction)
    p.location_type = getattr(
        dataclasses.I3Particle.LocationType, location_type
    )
    p.time = time
    if length is not None:
        p.length = length
    return p


def build_cc_event(case: "EdgeCase") -> "dataclasses.I3MCTree":
    """Neutrino created NU_DISTANCE upstream, CC-producing the charged lepton.

    The lepton takes the full energy (Bjorken y = 0), so there is no
    hadronic recoil at the vertex.
    """
    direction = _normalise(case.direction)
    vertex = np.array(case.position, dtype=float)
    nu_pos = vertex - direction * NU_DISTANCE

    neutrino = _make_particle(
        NEUTRINO_FOR[case.particle_type],
        case.energy,
        nu_pos,
        direction,
        length=NU_DISTANCE,
    )
    lepton = _make_particle(
        case.particle_type,
        case.energy,
        vertex,
        direction,
        time=NU_DISTANCE / C_M_PER_NS,
    )

    tree = dataclasses.I3MCTree()
    tree.add_primary(neutrino)
    tree.append_child(neutrino, lepton)
    return tree


def build_nan_primary_event(case: "EdgeCase") -> "dataclasses.I3MCTree":
    """CC event whose in-ice primary neutrino has NaN energy.

    A real NaN-energy primary carries a neutrino daughter with a defined
    energy, so ``check_primary_energy`` (used by ``I3HighestEParticleExtractor``)
    falls back from the NaN primary to that daughter. The primary is kept in-ice
    and at the top of the tree so it is the one ``get_primaries`` selects and the
    NaN branch is actually taken; the daughter neutrino then CCs inside the array
    to a real muon track.
    """
    direction = _normalise(case.direction)
    cc_vertex = np.array(case.position, dtype=float)
    nc_vertex = cc_vertex - direction * NC_TO_CC_DISTANCE
    nu_pos = nc_vertex - direction * NU_DISTANCE

    e_def = case.energy
    nu_flavour = NEUTRINO_FOR[case.particle_type]

    nu1 = _make_particle(
        nu_flavour, float("nan"), nu_pos, direction, length=NU_DISTANCE
    )
    t_nc = NU_DISTANCE / C_M_PER_NS
    hadrons_nc = _make_particle(
        "Hadrons", Y_NC * e_def, nc_vertex, direction, time=t_nc
    )
    e2 = (1.0 - Y_NC) * e_def
    nu2 = _make_particle(
        nu_flavour,
        e2,
        nc_vertex,
        direction,
        time=t_nc,
        length=NC_TO_CC_DISTANCE,
    )
    t_cc = t_nc + NC_TO_CC_DISTANCE / C_M_PER_NS
    hadrons_cc = _make_particle(
        "Hadrons", Y_CC * e2, cc_vertex, direction, time=t_cc
    )
    lepton = _make_particle(
        case.particle_type, (1.0 - Y_CC) * e2, cc_vertex, direction, time=t_cc
    )

    tree = dataclasses.I3MCTree()
    tree.add_primary(nu1)
    tree.append_child(nu1, hadrons_nc)
    tree.append_child(nu1, nu2)
    tree.append_child(nu2, hadrons_cc)
    tree.append_child(nu2, lepton)
    return tree


def build_outside_nc_event(case: "EdgeCase") -> "dataclasses.I3MCTree":
    """Primary neutrino interacts (NC) outside the ice; in-ice secondary CCs.

    The primary and its hadronic recoil are flagged non-InIce, so
    ``get_primaries`` cannot use the top-level primary and must recurse via
    ``find_in_ice_daughters`` to the in-ice secondary neutrino.
    """
    direction = _normalise(case.direction)
    cc_vertex = np.array(case.position, dtype=float)
    nc_vertex = cc_vertex - direction * NC_TO_CC_DISTANCE
    nu1_pos = nc_vertex - direction * NU_DISTANCE

    e1 = case.energy
    nu_flavour = NEUTRINO_FOR[case.particle_type]

    nu1 = _make_particle(
        nu_flavour,
        e1,
        nu1_pos,
        direction,
        length=NU_DISTANCE,
        location_type="Anywhere",
    )
    t_nc = NU_DISTANCE / C_M_PER_NS
    hadrons_nc = _make_particle(
        "Hadrons",
        Y_NC * e1,
        nc_vertex,
        direction,
        time=t_nc,
        location_type="Anywhere",
    )
    e2 = (1.0 - Y_NC) * e1
    nu2 = _make_particle(
        nu_flavour,
        e2,
        nc_vertex,
        direction,
        time=t_nc,
        length=NC_TO_CC_DISTANCE,
    )
    t_cc = t_nc + NC_TO_CC_DISTANCE / C_M_PER_NS
    hadrons_cc = _make_particle(
        "Hadrons", Y_CC * e2, cc_vertex, direction, time=t_cc
    )
    lepton = _make_particle(
        case.particle_type, (1.0 - Y_CC) * e2, cc_vertex, direction, time=t_cc
    )

    tree = dataclasses.I3MCTree()
    tree.add_primary(nu1)
    tree.append_child(nu1, hadrons_nc)
    tree.append_child(nu1, nu2)
    tree.append_child(nu2, hadrons_cc)
    tree.append_child(nu2, lepton)
    return tree


def build_coincident_background_event(
    case: "EdgeCase",
) -> "dataclasses.I3MCTree":
    """Signal neutrino misses the array, plus a coincident background muon.

    The neutrino's CC products (a low-energy muon and the hadronic recoil) all
    stop short of the hull, so the signal deposits nothing. An independent pi+
    decays to a muon that does cross the array, so the only charge in the
    detector comes from the coincident background -- the case the extractor
    represents as ``e_total == 0`` with ``daughters=True``.
    """
    direction = _normalise(case.direction)
    vertex = np.array(case.position, dtype=float)
    nu_pos = vertex - direction * NU_DISTANCE

    neutrino = _make_particle(
        NEUTRINO_FOR[case.particle_type],
        case.energy,
        nu_pos,
        direction,
        length=NU_DISTANCE,
    )
    t_cc = NU_DISTANCE / C_M_PER_NS
    hadrons = _make_particle(
        "Hadrons", Y_CC * case.energy, vertex, direction, time=t_cc
    )
    lepton = _make_particle(
        case.particle_type,
        (1.0 - Y_CC) * case.energy,
        vertex,
        direction,
        time=t_cc,
    )

    tree = dataclasses.I3MCTree()
    tree.add_primary(neutrino)
    tree.append_child(neutrino, hadrons)
    tree.append_child(neutrino, lepton)

    # Coincident background: a downgoing pi+ through the centre decays to the
    # muon that actually lights up the detector.
    bg_dir = _normalise((0.0, 0.0, -1.0))
    muon_pos = np.array([0.0, 0.0, DETECTOR_HALF_Z + 200.0])
    pion_pos = muon_pos - bg_dir * COINCIDENT_BG_PION_UPSTREAM
    pion = _make_particle(
        "PiPlus",
        COINCIDENT_BG_MUON_ENERGY,
        pion_pos,
        bg_dir,
        length=COINCIDENT_BG_PION_UPSTREAM,
        location_type="Anywhere",
    )
    muon = _make_particle(
        "MuPlus", COINCIDENT_BG_MUON_ENERGY, muon_pos, bg_dir
    )
    tree.add_primary(pion)
    tree.append_child(pion, muon)
    return tree


def build_corsika_bundle_event(case: "EdgeCase") -> "dataclasses.I3MCTree":
    """Cosmic-ray shower: a non-neutrino primary with a downgoing muon bundle.

    The extractor's ``is_corsika`` path treats the whole tree as background and
    sums the bundle from the unfiltered MMCTrackList. The bundle is illustrative
    -- multiplicity, energies and geometry are hand-chosen to exercise that path,
    not sampled from a flux model (for realistic bundles, use MuonGun). The muons
    are collinear with the shower axis and spread transversely around it, as a
    real in-ice bundle is.
    """
    direction = _normalise(case.direction)
    vertex = np.array(case.position, dtype=float)
    primary_pos = vertex - direction * CORSIKA_PRIMARY_UPSTREAM

    primary = _make_particle(
        case.particle_type,
        case.energy,
        primary_pos,
        direction,
        location_type="Anywhere",
    )
    tree = dataclasses.I3MCTree()
    tree.add_primary(primary)

    # Orthonormal vectors transverse to the shower axis, to place the muons on a
    # ring around it.
    ref = (
        np.array([1.0, 0.0, 0.0])
        if abs(direction[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(direction, ref)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)

    n = len(CORSIKA_BUNDLE_ENERGIES)
    for i, energy in enumerate(CORSIKA_BUNDLE_ENERGIES):
        if i == 0:
            offset = np.zeros(3)  # core muon on the shower axis
        else:
            phi = 2.0 * np.pi * (i - 1) / (n - 1)
            offset = CORSIKA_BUNDLE_RADIUS * (
                np.cos(phi) * u + np.sin(phi) * v
            )
        mu_pos = vertex + offset
        ptype = "MuMinus" if i % 2 == 0 else "MuPlus"
        muon = _make_particle(ptype, energy, mu_pos, direction)
        tree.append_child(primary, muon)
    return tree


# Directions are given as cartesian travel vectors (unambiguous, unlike the
# zenith/azimuth ctor) and normalised below. Energies are tuned so the muon
# range (~4-5 m/GeV) lands the stopping point where the topology wants it; the
# exact outcome depends on PROPOSAL and should be checked against the printed
# stats, then nudged if a case does not realise as intended.
EDGE_CASES: List[EdgeCase] = [
    EdgeCase(
        name="through_going_muon",
        particle_type="MuMinus",
        energy=1.0e5,
        position=(-800.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        comment="Enters from -x, crosses the centre, exits +x: e_dep < e_ent.",
    ),
    EdgeCase(
        name="stopping_track_contained",
        particle_type="MuMinus",
        energy=30.0,
        position=(-200.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        comment="Starts inside and ranges out inside: deposits ~all energy.",
    ),
    EdgeCase(
        name="starting_track",
        particle_type="MuMinus",
        energy=500.0,
        position=(-200.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        comment="Vertex inside the hull, exits: entrance == vertex energy.",
    ),
    EdgeCase(
        name="tau_to_mu_decay",
        particle_type="TauMinus",
        energy=1.0e7,
        position=(-200.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        comment="10 PeV tau (mean gamma*c*tau ~ 490 m) decays in-volume; ~17% "
        "to mu, so may need reruns/extra seeds to land the mu channel.",
    ),
    EdgeCase(
        name="nan_primary_energy",
        particle_type="MuMinus",
        energy=500.0,
        position=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        builder=build_nan_primary_event,
        comment="In-ice primary NuMu has NaN energy; its neutrino daughter "
        "carries the real energy, so check_primary_energy "
        "(I3HighestEParticleExtractor) falls back from the NaN primary to that "
        "daughter.",
    ),
    EdgeCase(
        name="non_top_level_in_ice_nu",
        particle_type="MuMinus",
        energy=1.0e4,
        position=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        builder=build_outside_nc_event,
        top_primary_outside_ice=True,
        comment="Primary NuMu flagged non-InIce interacts (NC) outside the "
        "ice; the in-ice secondary NuMu does CC at the centre, so the first "
        "in-ice neutrino is not at the top of the tree.",
    ),
    EdgeCase(
        name="coincident_background_muon",
        particle_type="MuMinus",
        energy=50.0,
        position=(-2000.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        builder=build_coincident_background_event,
        comment="Signal NuMu 2 km out whose muon ranges out and hadrons sit "
        "outside the array, plus a coincident pi+ -> muon that does cross it, "
        "so the only detector light is background (e_total == 0 with "
        "daughters=True).",
    ),
    EdgeCase(
        name="corsika_muon_bundle",
        particle_type="PPlus",
        energy=1.0e6,
        position=(0.0, 0.0, 800.0),
        direction=(0.0, 0.0, -1.0),
        builder=build_corsika_bundle_event,
        is_corsika=True,
        comment="Cosmic-ray air shower: a PPlus primary with a downgoing muon "
        "bundle. Extracted with is_corsika=True, where the whole tree is "
        "background and the bundle is summed from the unfiltered MMCTrackList.",
    ),
]


# --- Tray modules -----------------------------------------------------------


class SeedInjector(icetray.I3Module):
    """Inject one EdgeCase's pre-propagation primary per DAQ frame.

    Also stamps an I3EventHeader: it is required by I3NullSplitter to spawn the
    Physics frame, and several extractor error/warning paths read it.
    """

    def __init__(self, context: "icetray.I3Context") -> None:
        """Declare module parameters."""
        super().__init__(context)
        self.AddParameter("EdgeCases", "List of EdgeCase to inject", [])
        self._index = 0

    def Configure(self) -> None:
        """Read the configured edge cases."""
        self._edge_cases = self.GetParameter("EdgeCases")

    def DAQ(self, frame: "icetray.I3Frame") -> None:
        """Inject one edge case's primary, or suspend when exhausted."""
        # I3InfiniteSource emits DAQ frames endlessly; stop once every edge
        # case has been injected rather than assuming a fixed GCD-frame count.
        if self._index >= len(self._edge_cases):
            self.RequestSuspension()
            return
        case = self._edge_cases[self._index]

        builder = case.builder or build_cc_event
        frame["I3MCTree_preMuonProp"] = builder(case)

        header = dataclasses.I3EventHeader()
        header.run_id = 0
        header.event_id = self._index
        frame["I3EventHeader"] = header

        frame["EdgeCaseName"] = dataclasses.I3String(case.name)
        frame["TopPrimaryOutsideIce"] = icetray.I3Bool(
            case.top_primary_outside_ice
        )
        frame["IsCorsika"] = icetray.I3Bool(case.is_corsika)

        self._index += 1
        self.PushFrame(frame)


def ensure_mmctracklist(frame: "icetray.I3Frame") -> bool:
    """Attach an empty MMCTrackList when propagation produced none.

    I3PropagatorModule omits the key when no muon is tracked (the no-
    reach case), but the validator and extractor both expect it present,
    as the full PropagateMuons segment would write it.
    """
    if "MMCTrackList" not in frame:
        frame["MMCTrackList"] = simclasses.I3MMCTrackList()
    return True


def _point_inside(pos: "dataclasses.I3Position") -> bool:
    """Cylinder approximation of containment, for the printed stats only."""
    return (
        np.hypot(pos.x, pos.y) < DETECTOR_RADIUS
        and abs(pos.z) < DETECTOR_HALF_Z
    )


def describe_topology(frame: "icetray.I3Frame") -> str:
    """One-line summary of what a frame's MMCTracks actually look like."""
    tracks = MuonGun.Track.harvest(frame["I3MCTree"], frame["MMCTrackList"])
    parts = []
    for track in tracks:
        start_in = _point_inside(track.pos)
        end_pos = dataclasses.I3Position(
            track.pos.x + track.dir.x * track.length,
            track.pos.y + track.dir.y * track.length,
            track.pos.z + track.dir.z * track.length,
        )
        end_in = _point_inside(end_pos)
        parts.append(
            f"{track.type}(E={track.energy:.1f}GeV, len={track.length:.0f}m, "
            f"start_in={start_in}, end_in={end_in})"
        )
    return "; ".join(parts) if parts else "<no harvested tracks>"


class FrameValidator(icetray.I3Module):
    """Assert the tree and MMCTrackList are present and self-consistent.

    Runs on the stream given by ``Stream`` so the same checks can guard both the
    DAQ frame (where the objects live) and the Physics frame (where the
    extractor reads them, via frame mixing).
    """

    def __init__(self, context: "icetray.I3Context") -> None:
        """Declare validation parameters."""
        super().__init__(context)
        self.AddParameter("MCTreeName", "Post-prop tree key", "I3MCTree")
        self.AddParameter("MMCTrackListName", "Track list key", "MMCTrackList")
        self.AddParameter(
            "Stream", "Frame stop to validate", icetray.I3Frame.DAQ
        )
        self.AddParameter("Label", "Tag for log messages", "")

    def Configure(self) -> None:
        """Read the configured parameters."""
        self._mctree = self.GetParameter("MCTreeName")
        self._mmc = self.GetParameter("MMCTrackListName")
        self._stream = self.GetParameter("Stream")
        self._label = self.GetParameter("Label")

    def _validate(self, frame: "icetray.I3Frame") -> None:
        name = (
            str(frame["EdgeCaseName"].value)
            if frame.Has("EdgeCaseName")
            else "<unknown>"
        )
        ctx = f"[{self._label}] event '{name}'"

        if not frame.Has(self._mctree):
            raise RuntimeError(f"{ctx}: missing tree '{self._mctree}'")
        if not frame.Has(self._mmc):
            raise RuntimeError(f"{ctx}: missing track list '{self._mmc}'")

        tree = frame[self._mctree]
        mmc_list = frame[self._mmc]

        if len(tree) == 0:
            raise RuntimeError(f"{ctx}: '{self._mctree}' is empty")

        if len(mmc_list) == 0:
            raise RuntimeError(
                f"{ctx}: '{self._mmc}' is empty (no propagated track -- "
                "retune the seed energy/geometry for this case)"
            )

        # Every MMCTrack must point at a particle that exists in the tree; this
        # is exactly the invariant filter_track_list/harvest depend on.
        for mmc_track in mmc_list:
            pid = mmc_track.particle.id
            try:
                tree.get_particle(pid)
            except RuntimeError:
                raise RuntimeError(
                    f"{ctx}: MMCTrack particle {pid} absent from "
                    f"'{self._mctree}' -- tree/track-list inconsistent"
                )

        # The extractor's hard dependency: harvest must succeed and yield a
        # track whose energy is queryable along its length.
        harvested = MuonGun.Track.harvest(tree, mmc_list)
        if len(harvested) == 0:
            raise RuntimeError(
                f"{ctx}: MuonGun.Track.harvest returned no tracks"
            )
        for track in harvested:
            _ = track.get_energy(0.0)

        if (
            frame.Has("TopPrimaryOutsideIce")
            and frame["TopPrimaryOutsideIce"].value
        ):
            self._assert_in_ice_nu_below_top(tree, ctx)

        print(f"{ctx}: OK -- {describe_topology(frame)}")

    def _assert_in_ice_nu_below_top(
        self, tree: "dataclasses.I3MCTree", ctx: str
    ) -> None:
        """Fail unless the first in-ice neutrino is below the top of the tree.

        ``get_primaries`` only recurses when no top-level primary is an in-ice
        neutrino; assert it so the case cannot silently take the ordinary path.
        """
        in_ice = dataclasses.I3Particle.LocationType.InIce
        top_ids = {p.id for p in tree.get_primaries()}
        in_ice_nus = [
            p for p in tree if p.is_neutrino and p.location_type == in_ice
        ]
        has_top_in_ice_nu = any(p.id in top_ids for p in in_ice_nus)
        has_below_in_ice_nu = any(p.id not in top_ids for p in in_ice_nus)
        if has_top_in_ice_nu or not has_below_in_ice_nu:
            raise RuntimeError(
                f"{ctx}: expected the first in-ice neutrino below the top of "
                "the tree so get_primaries recurses (top-level in-ice nu: "
                f"{has_top_in_ice_nu}, deeper in-ice nu: {has_below_in_ice_nu})"
            )

    def DAQ(self, frame: "icetray.I3Frame") -> None:
        """Validate the DAQ frame when configured for that stream."""
        if self._stream == icetray.I3Frame.DAQ:
            self._validate(frame)
        self.PushFrame(frame)

    def Physics(self, frame: "icetray.I3Frame") -> None:
        """Validate the Physics frame when configured for that stream."""
        if self._stream == icetray.I3Frame.Physics:
            self._validate(frame)
        self.PushFrame(frame)


# --- Driver -----------------------------------------------------------------


def prepare_proposal_config() -> str:
    """Write a PROPOSAL config whose interpolation tables are writable.

    Returns the path to the patched config to hand to PropagateMuons.
    """
    with open(PROPOSAL_CONFIG_IN) as fh:
        config = json.load(fh)

    os.makedirs(PROPOSAL_TABLES_DIR, exist_ok=True)
    interpolation = config["global"]["interpolation"]
    interpolation["path_to_tables"] = [PROPOSAL_TABLES_DIR]
    interpolation["path_to_tables_readonly"] = [PROPOSAL_TABLES_DIR]
    interpolation["just_use_readonly_path"] = False

    with open(PROPOSAL_CONFIG_OUT, "w") as fh:
        json.dump(config, fh, indent=2)
    return PROPOSAL_CONFIG_OUT


def main() -> None:
    """Generate, validate, and write the i3 fixture events."""
    if not os.path.exists(GCD_FILE):
        raise FileNotFoundError(
            f"GCD not found at {GCD_FILE}. Set $I3_TESTDATA (it ships with "
            "icetray's test data) or point GCD_FILE at a readable GCD."
        )

    proposal_config = prepare_proposal_config()
    rng = phys_services.I3GSLRandomService(seed=RNG_SEED)

    tray = icetray.I3Tray()
    tray.Add("I3InfiniteSource", Prefix=GCD_FILE)
    tray.Add(SeedInjector, "seed", EdgeCases=EDGE_CASES)

    # Build only the gen1 muon/tau propagators. The PropagateMuons segment also
    # constructs a Gen2 propagator from a stock, read-only config whose tables
    # cannot be built in this image; our non-Gen2 events never use it, so we
    # drive I3PropagatorModule directly to avoid that path entirely.
    propagators = make_standard_propagators(
        PROPOSAL_config_file=proposal_config
    )
    tray.AddModule(
        "I3PropagatorModule",
        "propagate",
        PropagatorServices=propagators,
        RandomService=rng,
        InputMCTreeName="I3MCTree_preMuonProp",
        OutputMCTreeName="I3MCTree",
        RNGStateName="I3MCTree_preMuonProp_RNGState",
    )
    tray.AddModule(
        ensure_mmctracklist,
        "ensure_mmctracklist",
        Streams=[icetray.I3Frame.DAQ],
    )

    # Validate where the objects physically live (DAQ)...
    tray.Add(
        FrameValidator,
        "validate_daq",
        Stream=icetray.I3Frame.DAQ,
        Label="DAQ",
    )

    # ...then materialise a Physics frame per event and validate what the
    # extractor will actually see there (Q content reaches P via frame mixing).
    tray.Add("I3NullSplitter", "nullsplit", SubEventStreamName="fullevent")
    tray.Add(
        FrameValidator,
        "validate_physics",
        Stream=icetray.I3Frame.Physics,
        Label="Physics",
    )

    tray.Add("I3Writer", "writer", filename=OUTPUT_FILE)

    # SeedInjector suspends the tray once all edge cases are injected, so run
    # unbounded rather than guessing the GCD-frame count.
    tray.Execute()

    print(f"\nWrote {len(EDGE_CASES)} validated events to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
