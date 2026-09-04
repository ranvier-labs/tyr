import Tyr.EventSkeleton.Contact
import Tyr.EventSkeleton.Physics

/-!
# Tyr.EventSkeleton.SceneGraph

SceneGraph-style geometry provider primitives for event-skeleton examples.

This module is intentionally a provider boundary, not a full collision engine.
It stores sources, frames, geometries, roles, poses, and material metadata in a
form close to Drake's `SceneGraph`.  Downstream physics code can ask the
provider for primitive contact views such as `ContactCandidate`; a richer
broadphase, narrowphase, hydroelastic backend, or URDF/SDF importer can replace
the internals without changing the event/contact moves.
-/

namespace Tyr.EventSkeleton

structure SceneVec3 where
  x : Float := 0.0
  y : Float := 0.0
  z : Float := 0.0
  deriving Repr, BEq, Inhabited

namespace SceneVec3

def zero : SceneVec3 := {}

def unitX : SceneVec3 := { x := 1.0 }
def unitY : SceneVec3 := { y := 1.0 }
def unitZ : SceneVec3 := { z := 1.0 }

def add (a b : SceneVec3) : SceneVec3 :=
  { x := a.x + b.x, y := a.y + b.y, z := a.z + b.z }

def sub (a b : SceneVec3) : SceneVec3 :=
  { x := a.x - b.x, y := a.y - b.y, z := a.z - b.z }

def scale (s : Float) (v : SceneVec3) : SceneVec3 :=
  { x := s * v.x, y := s * v.y, z := s * v.z }

def dot (a b : SceneVec3) : Float :=
  a.x * b.x + a.y * b.y + a.z * b.z

def cross (a b : SceneVec3) : SceneVec3 :=
  {
    x := a.y * b.z - a.z * b.y
    y := a.z * b.x - a.x * b.z
    z := a.x * b.y - a.y * b.x
  }

def norm (v : SceneVec3) : Float :=
  Float.sqrt (v.dot v)

def normalize? (v : SceneVec3) (label : String := "vector") : Except String SceneVec3 := do
  let n := v.norm
  if !(Float.isFinite n) || n < 1.0e-12 then
    .error s!"{label}: cannot normalize vector with norm {n}"
  else
    pure (v.scale (1.0 / n))

def toArray (v : SceneVec3) : Array Float :=
  #[v.x, v.y, v.z]

def isFinite (v : SceneVec3) : Bool :=
  Float.isFinite v.x && Float.isFinite v.y && Float.isFinite v.z

end SceneVec3

structure SceneRgba where
  r : Float := 0.0
  g : Float := 0.0
  b : Float := 0.0
  a : Float := 1.0
  deriving Repr, BEq, Inhabited

namespace SceneRgba

def validate? (rgba : SceneRgba) (label : String := "rgba") : Except String Unit := do
  let values := #[rgba.r, rgba.g, rgba.b, rgba.a]
  for i in [:values.size] do
    let x := values[i]!
    if !(Float.isFinite x) || x < 0.0 || x > 1.0 then
      .error s!"{label}: channel {i} must lie in [0, 1], got {x}"

end SceneRgba

/-- Geometry roles mirror Drake's illustration/perception/proximity split. -/
inductive SceneGeometryRole where
  | illustration
  | perception
  | proximity
  deriving Repr, BEq, Inhabited

/-- Minimal hydroelastic metadata attached to proximity geometry. -/
inductive SceneHydroelasticProperty where
  | rigid (resolutionHint : Float)
  | compliant (resolutionHint : Float) (hydroelasticModulus : Float)
  deriving Repr, BEq, Inhabited

namespace SceneHydroelasticProperty

def validate? (property : SceneHydroelasticProperty) (label : String := "hydroelastic") :
    Except String Unit := do
  match property with
  | .rigid h =>
      if !(Float.isFinite h) || h <= 0.0 then
        .error s!"{label}: rigid resolution hint must be positive and finite, got {h}"
  | .compliant h modulus =>
      if !(Float.isFinite h) || h <= 0.0 then
        .error s!"{label}: compliant resolution hint must be positive and finite, got {h}"
      if !(Float.isFinite modulus) || modulus <= 0.0 then
        .error s!"{label}: compliant modulus must be positive and finite, got {modulus}"

end SceneHydroelasticProperty

inductive SceneGeometryShape where
  | sphere (radius : Float)
  | halfSpace (normal : SceneVec3) (point : SceneVec3)
  | box (sizeX sizeY sizeZ : Float)
  | cylinder (radius length : Float)
  | capsule (radius length : Float)
  | model (uri : String)
  | mesh (uri : String) (scale : Float) (supportingFiles : Array String)
  | convex (uri : String) (scale : Float)
  deriving Repr, BEq, Inhabited

namespace SceneGeometryShape

private def finitePositive (x : Float) : Bool :=
  Float.isFinite x && x > 0.0

def validate? (shape : SceneGeometryShape) (label : String := "shape") :
    Except String Unit := do
  match shape with
  | .sphere radius =>
      if !finitePositive radius then
        .error s!"{label}: sphere radius must be positive and finite, got {radius}"
  | .halfSpace normal point =>
      let _ ← normal.normalize? s!"{label}: half-space normal"
      if !point.isFinite then
        .error s!"{label}: half-space point must be finite"
  | .box sx sy sz =>
      if !finitePositive sx || !finitePositive sy || !finitePositive sz then
        .error s!"{label}: box dimensions must be positive and finite"
  | .cylinder radius length =>
      if !finitePositive radius || !finitePositive length then
        .error s!"{label}: cylinder radius and length must be positive and finite"
  | .capsule radius length =>
      if !finitePositive radius || !finitePositive length then
        .error s!"{label}: capsule radius and length must be positive and finite"
  | .model uri =>
      if uri.isEmpty then
        .error s!"{label}: model uri cannot be empty"
  | .mesh uri scale _ =>
      if uri.isEmpty then
        .error s!"{label}: mesh uri cannot be empty"
      if !finitePositive scale then
        .error s!"{label}: mesh scale must be positive and finite, got {scale}"
  | .convex uri scale =>
      if uri.isEmpty then
        .error s!"{label}: convex mesh uri cannot be empty"
      if !finitePositive scale then
        .error s!"{label}: convex scale must be positive and finite, got {scale}"

def name : SceneGeometryShape → String
  | .sphere _ => "sphere"
  | .halfSpace _ _ => "half_space"
  | .box _ _ _ => "box"
  | .cylinder _ _ => "cylinder"
  | .capsule _ _ => "capsule"
  | .model _ => "model"
  | .mesh _ _ _ => "mesh"
  | .convex _ _ => "convex"

end SceneGeometryShape

/--
Rigid pose with the rotation stored as an axis-angle pair.

Only simple pose metadata is needed for the current event-skeleton examples;
geometry providers can replace this with quaternions or full transforms later.
-/
structure ScenePose3 where
  translation : SceneVec3 := {}
  rotationAxis : SceneVec3 := SceneVec3.unitZ
  rotationAngle : Float := 0.0
  deriving Repr, BEq, Inhabited

namespace ScenePose3

def identity : ScenePose3 := {}

def translated (p : SceneVec3) : ScenePose3 :=
  { translation := p }

private def clampUnit (x : Float) : Float :=
  min 1.0 (max (-1.0) x)

def fromAxisAngle (translation axis : SceneVec3) (angle : Float) : ScenePose3 :=
  { translation := translation, rotationAxis := axis, rotationAngle := angle }

/--
Convert Drake-style roll-pitch-yaw angles to the axis-angle storage used by
`ScenePose3`.  The rotation convention is `Rz(yaw) * Ry(pitch) * Rx(roll)`.
-/
def fromRollPitchYaw (translation : SceneVec3) (roll pitch yaw : Float) : ScenePose3 :=
  let hr := 0.5 * roll
  let hp := 0.5 * pitch
  let hy := 0.5 * yaw
  let cr := Float.cos hr
  let sr := Float.sin hr
  let cp := Float.cos hp
  let sp := Float.sin hp
  let cy := Float.cos hy
  let sy := Float.sin hy
  let qw := cr * cp * cy + sr * sp * sy
  let qx := sr * cp * cy - cr * sp * sy
  let qy := cr * sp * cy + sr * cp * sy
  let qz := cr * cp * sy - sr * sp * cy
  let angle := 2.0 * Float.acos (clampUnit qw)
  let s := Float.sqrt (max 0.0 (1.0 - qw * qw))
  if s < 1.0e-12 then
    { translation := translation, rotationAxis := SceneVec3.unitZ, rotationAngle := 0.0 }
  else
    {
      translation := translation
      rotationAxis := { x := qx / s, y := qy / s, z := qz / s }
      rotationAngle := angle
    }

def rotateVector (pose : ScenePose3) (v : SceneVec3) : SceneVec3 :=
  let n := pose.rotationAxis.norm
  if !(Float.isFinite n) || n < 1.0e-12 then
    v
  else
    let axis := pose.rotationAxis.scale (1.0 / n)
    let c := Float.cos pose.rotationAngle
    let s := Float.sin pose.rotationAngle
    let cross := axis.cross v
    let dot := axis.dot v
    (v.scale c).add ((cross.scale s).add (axis.scale (dot * (1.0 - c))))

def validate? (pose : ScenePose3) (label : String := "pose") : Except String Unit := do
  if !pose.translation.isFinite then
    .error s!"{label}: translation must be finite"
  let _ ← pose.rotationAxis.normalize? s!"{label}: rotation axis"
  if !(Float.isFinite pose.rotationAngle) then
    .error s!"{label}: rotation angle must be finite, got {pose.rotationAngle}"

end ScenePose3

structure SceneGeometryProperties where
  roles : Array SceneGeometryRole := #[]
  diffuseRgba? : Option SceneRgba := none
  renderLabel? : Option Nat := none
  friction : CoulombFriction := CoulombFriction.frictionless
  hydroelastic? : Option SceneHydroelasticProperty := none
  deriving Repr, BEq, Inhabited

namespace SceneGeometryProperties

def hasRole (properties : SceneGeometryProperties) (role : SceneGeometryRole) : Bool :=
  properties.roles.any (fun r => r == role)

def validate? (properties : SceneGeometryProperties) (label : String := "geometry properties") :
    Except String Unit := do
  match properties.diffuseRgba? with
  | some rgba => rgba.validate? s!"{label}: diffuse"
  | none => pure ()
  properties.friction.validate? s!"{label}: friction"
  match properties.hydroelastic? with
  | some h =>
      if !properties.hasRole .proximity then
        .error s!"{label}: hydroelastic metadata requires proximity role"
      h.validate? s!"{label}: hydroelastic"
  | none => pure ()

end SceneGeometryProperties

structure SceneSource where
  id : Nat
  name : String
  deriving Repr, BEq, Inhabited

structure SceneFrame where
  id : Nat
  sourceId : Nat
  name : String
  parentFrameId? : Option Nat := none
  frameGroup : Nat := 0
  deriving Repr, BEq, Inhabited

structure SceneGeometry where
  id : Nat
  sourceId : Nat
  frameId? : Option Nat := none
  X_FG : ScenePose3 := {}
  shape : SceneGeometryShape := .sphere 1.0
  name : String := ""
  properties : SceneGeometryProperties := { roles := #[] }
  deriving Repr, BEq, Inhabited

namespace SceneGeometry

def isAnchored (geometry : SceneGeometry) : Bool :=
  geometry.frameId?.isNone

def hasRole (geometry : SceneGeometry) (role : SceneGeometryRole) : Bool :=
  geometry.properties.hasRole role

def validate? (geometry : SceneGeometry) : Except String Unit := do
  if geometry.name.isEmpty then
    .error s!"geometry {geometry.id}: name cannot be empty"
  geometry.X_FG.validate? s!"geometry {geometry.id} pose"
  geometry.shape.validate? s!"geometry {geometry.id} shape"
  geometry.properties.validate? s!"geometry {geometry.id} properties"

end SceneGeometry

structure SceneFramePose where
  frameId : Nat
  X_WF : ScenePose3
  deriving Repr, BEq, Inhabited

structure SceneFramePoseVector where
  poses : Array SceneFramePose := #[]
  deriving Repr, BEq, Inhabited

structure SceneGraphProvider where
  worldFrameId : Nat := 0
  sources : Array SceneSource := #[]
  frames : Array SceneFrame := #[]
  geometries : Array SceneGeometry := #[]
  label : String := ""
  deriving Repr, BEq, Inhabited

namespace SceneGraphProvider

private def containsNat (needle : Nat) (xs : Array Nat) : Bool :=
  xs.any (fun x => x == needle)

private def hasDuplicateNat (xs : Array Nat) : Bool := Id.run do
  let mut seen : Array Nat := #[]
  for x in xs do
    if containsNat x seen then
      return true
    seen := seen.push x
  return false

def sourceIds (provider : SceneGraphProvider) : Array Nat :=
  provider.sources.map (fun source => source.id)

def frameIds (provider : SceneGraphProvider) : Array Nat :=
  provider.frames.map (fun frame => frame.id)

def geometryIds (provider : SceneGraphProvider) : Array Nat :=
  provider.geometries.map (fun geometry => geometry.id)

def sourceById? (provider : SceneGraphProvider) (id : Nat) : Option SceneSource :=
  provider.sources.find? (fun source => source.id == id)

def frameById? (provider : SceneGraphProvider) (id : Nat) : Option SceneFrame :=
  if id == provider.worldFrameId then
    some { id := provider.worldFrameId, sourceId := 0, name := "world" }
  else
    provider.frames.find? (fun frame => frame.id == id)

def geometryById? (provider : SceneGraphProvider) (id : Nat) : Option SceneGeometry :=
  provider.geometries.find? (fun geometry => geometry.id == id)

def geometriesWithRole
    (provider : SceneGraphProvider) (role : SceneGeometryRole) : Array SceneGeometry :=
  provider.geometries.filter (fun geometry => geometry.hasRole role)

def numGeometriesForFrameWithRole
    (provider : SceneGraphProvider) (frameId : Nat) (role : SceneGeometryRole) : Nat :=
  (provider.geometries.filter
    (fun geometry => geometry.frameId? == some frameId && geometry.hasRole role)).size

def anchoredGeometries (provider : SceneGraphProvider) : Array SceneGeometry :=
  provider.geometries.filter (fun geometry => geometry.isAnchored)

def validate? (provider : SceneGraphProvider) : Except String Unit := do
  if hasDuplicateNat provider.sourceIds then
    .error s!"scene graph {provider.label}: duplicate source id"
  if hasDuplicateNat provider.frameIds then
    .error s!"scene graph {provider.label}: duplicate frame id"
  if hasDuplicateNat provider.geometryIds then
    .error s!"scene graph {provider.label}: duplicate geometry id"
  for source in provider.sources do
    if source.name.isEmpty then
      .error s!"scene graph {provider.label}: source {source.id} has empty name"
  for frame in provider.frames do
    if frame.name.isEmpty then
      .error s!"scene graph {provider.label}: frame {frame.id} has empty name"
    if (provider.sourceById? frame.sourceId).isNone then
      .error s!"scene graph {provider.label}: frame {frame.id} references missing source {frame.sourceId}"
    match frame.parentFrameId? with
    | some parent =>
        if parent != provider.worldFrameId && (provider.frameById? parent).isNone then
          .error s!"scene graph {provider.label}: frame {frame.id} references missing parent frame {parent}"
    | none => pure ()
  for geometry in provider.geometries do
    geometry.validate?
    if (provider.sourceById? geometry.sourceId).isNone then
      .error s!"scene graph {provider.label}: geometry {geometry.id} references missing source {geometry.sourceId}"
    match geometry.frameId? with
    | some frameId =>
        if frameId != provider.worldFrameId && (provider.frameById? frameId).isNone then
          .error s!"scene graph {provider.label}: geometry {geometry.id} references missing frame {frameId}"
    | none => pure ()

def shapeNames (provider : SceneGraphProvider) : Array String :=
  provider.geometries.map (fun geometry => geometry.shape.name)

end SceneGraphProvider

namespace SceneFramePoseVector

def poseForFrame? (poses : SceneFramePoseVector) (frameId : Nat) : Option ScenePose3 :=
  (poses.poses.find? (fun pose => pose.frameId == frameId)).map (fun pose => pose.X_WF)

def validate? (poses : SceneFramePoseVector) (provider : SceneGraphProvider) :
    Except String Unit := do
  for pose in poses.poses do
    if (provider.frameById? pose.frameId).isNone then
      .error s!"frame pose vector references missing frame {pose.frameId}"
    pose.X_WF.validate? s!"frame pose {pose.frameId}"

end SceneFramePoseVector

structure ScenePointPairPenetration where
  idA : Nat
  idB : Nat
  depth : Float
  nhatBA_W : SceneVec3 := SceneVec3.unitZ
  p_WCa : SceneVec3 := {}
  p_WCb : SceneVec3 := {}
  label : String := ""
  deriving Repr, BEq, Inhabited

namespace ScenePointPairPenetration

def validate? (pair : ScenePointPairPenetration) : Except String Unit := do
  if !(Float.isFinite pair.depth) || pair.depth < 0.0 then
    .error s!"point-pair penetration {pair.label}: depth must be nonnegative and finite, got {pair.depth}"
  let _ ← pair.nhatBA_W.normalize? s!"point-pair penetration {pair.label}: normal"
  if !pair.p_WCa.isFinite || !pair.p_WCb.isFinite then
    .error s!"point-pair penetration {pair.label}: witness points must be finite"

end ScenePointPairPenetration

/--
Dynamic SceneGraph contact query output.

This is the provider-side boundary before support selection.  A full collision
engine can compute hydroelastic surfaces, point-pair fallback contacts, and
solver-facing candidate views in its own storage layout, then expose just these
primitive products to the event-skeleton physics layer.
-/
structure SceneContactQueryResult where
  providerLabel : String := ""
  hydroelasticPatches : Array HydroelasticContactPatch := #[]
  pointPairs : Array ScenePointPairPenetration := #[]
  candidates : ContactCandidateSet := {}
  useStrictHydro : Bool := true
  representation : HydroelasticSurfaceRepresentation := .triangle
  label : String := ""
  deriving Repr, Inhabited

namespace SceneContactQueryResult

def totalPrimitiveContacts (result : SceneContactQueryResult) : Nat :=
  result.hydroelasticPatches.size + result.pointPairs.size + result.candidates.candidates.size

def hasFallbackPointPairs (result : SceneContactQueryResult) : Bool :=
  !result.pointPairs.isEmpty

/--
Project a dynamic SceneGraph query to the contact-candidate primitive consumed
by the full-physics layer.

Point-pair fallback records alone are not enough for dynamics because they lack
generalized-velocity Jacobian rows.  A provider that uses point pairs for
collision must also expose corresponding `ContactCandidate` views before the
mass-matrix primitive can consume the contact.
-/
def solverContactCandidateSet? (result : SceneContactQueryResult) :
    Except String ContactCandidateSet := do
  if !result.candidates.candidates.isEmpty then
    result.candidates.validate?
    pure result.candidates
  else if !result.pointPairs.isEmpty then
    .error s!"scene contact query {result.label}: point-pair fallback contacts need provider-specific ContactCandidate views with generalized-velocity Jacobian rows"
  else if !result.hydroelasticPatches.isEmpty then
    let candidates := result.hydroelasticPatches.map
      (fun patch => patch.equivalentContactCandidate)
    pure {
      candidates := candidates
      sourceCandidateCount? := some result.hydroelasticPatches.size
      label := s!"hydroelastic-solver-contact-candidates:{result.label}"
    }
  else
    pure {
      candidates := #[]
      sourceCandidateCount? := some 0
      label := s!"empty-solver-contact-candidates:{result.label}"
    }

def solverContactSupport?
    (result : SceneContactQueryResult)
    (policy : SupportPolicy := .fullSupport) :
    Except String ContactSupport := do
  let candidates ← result.solverContactCandidateSet?
  pure (candidates.selectWithPolicy policy result.label)

def hydroelasticSupport (result : SceneContactQueryResult)
    (minArea : Float := 0.0) : HydroelasticPatchSupport :=
  HydroelasticPatchSupport.selectByArea minArea result.hydroelasticPatches result.label

def contactSupport
    (result : SceneContactQueryResult)
    (policy : SupportPolicy := .fullSupport) : ContactSupport :=
  result.candidates.selectWithPolicy policy result.label

def validate? (result : SceneContactQueryResult) (jacobianWidth? : Option Nat := none) :
    Except String Unit := do
  for patch in result.hydroelasticPatches do
    patch.validateGeometry?
    match jacobianWidth? with
    | some width => patch.validateJacobianWidth? width
    | none => pure ()
  for pair in result.pointPairs do
    pair.validate?
  result.candidates.validate? jacobianWidth?
  if result.useStrictHydro && result.hasFallbackPointPairs then
    .error s!"scene contact query {result.label}: strict hydro query cannot include point-pair fallback contacts"

end SceneContactQueryResult

private def requireGeometry?
    (provider : SceneGraphProvider) (id : Nat) : Except String SceneGeometry := do
  match provider.geometryById? id with
  | some geometry => pure geometry
  | none => .error s!"scene graph {provider.label}: missing geometry {id}"

private def requireProximity? (geometry : SceneGeometry) : Except String Unit := do
  if !geometry.hasRole .proximity then
    .error s!"geometry {geometry.id} ({geometry.name}) does not have proximity role"

/--
Emit a primitive point-contact candidate for a sphere against a half-space.

This is a narrow provider view: the geometry registry owns shape/role metadata,
while the contact solver receives only signed distance, velocity, and `J` rows.
-/
def sphereHalfSpaceContactCandidate?
    (provider : SceneGraphProvider)
    (sphereGeometryId halfSpaceGeometryId : Nat)
    (sphereCenter_W : SceneVec3)
    (normalVelocity : Float)
    (normalJacobian tangentJacobian : Array Float)
    (tangentJacobian2 : Array Float := #[])
    (label : String := "") : Except String ContactCandidate := do
  let sphere ← requireGeometry? provider sphereGeometryId
  let halfSpace ← requireGeometry? provider halfSpaceGeometryId
  requireProximity? sphere
  requireProximity? halfSpace
  let radius ←
    match sphere.shape with
    | .sphere radius => pure radius
    | other => .error s!"geometry {sphereGeometryId} is {other.name}, expected sphere"
  let (normal, point) ←
    match halfSpace.shape with
    | .halfSpace normal point => pure (normal, point)
    | other => .error s!"geometry {halfSpaceGeometryId} is {other.name}, expected half_space"
  let nhat ← normal.normalize? s!"geometry {halfSpaceGeometryId} half-space normal"
  let signedDistance := nhat.dot (sphereCenter_W.sub point) - radius
  let contactPoint := SceneVec3.sub sphereCenter_W (SceneVec3.scale radius nhat)
  pure {
    id := sphereGeometryId * 100000 + halfSpaceGeometryId
    bodyA := sphere.name
    bodyB := halfSpace.name
    point_W := contactPoint.toArray
    normal_W := nhat.toArray
    signedDistance := signedDistance
    normalVelocity := normalVelocity
    tangentVelocity := 0.0
    tangentVelocity2 := 0.0
    normalJacobian := normalJacobian
    tangentJacobian := tangentJacobian
    tangentJacobian2 := tangentJacobian2
    label :=
      if label.isEmpty then
        s!"{sphere.name}-{halfSpace.name}"
      else
        label
  }

def sphereHalfSpacePenetration?
    (provider : SceneGraphProvider)
    (sphereGeometryId halfSpaceGeometryId : Nat)
    (sphereCenter_W : SceneVec3)
    (label : String := "") : Except String (Option ScenePointPairPenetration) := do
  let candidate ← sphereHalfSpaceContactCandidate?
    provider sphereGeometryId halfSpaceGeometryId sphereCenter_W
    0.0 #[1.0] #[0.0] #[] label
  if candidate.signedDistance < 0.0 then
    let depth := -candidate.signedDistance
    pure (some {
      idA := sphereGeometryId
      idB := halfSpaceGeometryId
      depth := depth
      nhatBA_W := SceneVec3.unitZ
      p_WCa := { sphereCenter_W with z := sphereCenter_W.z - depth }
      p_WCb := { sphereCenter_W with z := 0.0 }
      label := candidate.label
    })
  else
    pure none

end Tyr.EventSkeleton
