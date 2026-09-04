import Tyr.DiffEq.Integrate
import Tyr.DiffEq.Solver.RK4
import Tyr.DiffEq.Term
import Tyr.EventSkeleton.Manipulator
import Tyr.EventSkeleton.SceneGraph
import Tyr.EventSkeleton.Trace

/-!
# Drake Quadrotor Event-Skeleton Example

This ports the explicit 12-state rigid-body dynamics from
`../drake/examples/quadrotor/quadrotor_plant.{h,cc}`.

The state order follows Drake's `QuadrotorPlant`:

`[x, y, z, roll, pitch, yaw, xdot, ydot, zdot, rolldot, pitchdot, yawdot]`.

The implementation keeps the rigid-body computation local to this example:
rotor thrusts are summed in the body frame, transformed to world coordinates by
roll-pitch-yaw, and angular acceleration is computed from Euler's rigid-body
equation with diagonal inertia.  The resulting derivative is integrated through
the same ODE/event-skeleton trace path as the other Drake ports.
-/

namespace Tyr.EventSkeleton.Examples.Quadrotor

open Tyr.EventSkeleton
open torch.DiffEq

abbrev QuadrotorState := Array Float
abbrev QuadrotorInput := Array Float

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/quadrotor/quadrotor_plant.cc"
      concept := "implements the explicit 12-state QuadrotorPlant dynamics"
    },
    {
      path := "../drake/examples/quadrotor/quadrotor_plant.h"
      concept := "declares default mass, arm length, force/moment constants, and input/state ports"
    },
    {
      path := "../drake/examples/quadrotor/test/quadrotor_plant_test.cc"
      concept := "checks disconnected-input derivative behavior and scalar conversion"
    },
    {
      path := "../drake/examples/quadrotor/test/quadrotor_dynamics_test.cc"
      concept := "compares 12-state QuadrotorPlant dynamics against a MultibodyPlant URDF model"
    },
    {
      path := "../drake/examples/quadrotor/run_quadrotor_dynamics.cc"
      concept := "runs the explicit QuadrotorPlant dynamics with visualization"
    },
    {
      path := "../drake/examples/quadrotor/run_quadrotor_lqr.cc"
      concept := "builds the QuadrotorPlant, StabilizingLQRController, SceneGraph, and seven-second hover trials"
    },
    {
      path := "../drake/examples/quadrotor/warehouse.sdf"
      concept := "static warehouse model loaded by run_quadrotor_dynamics as floor, walls, and slalom obstacles"
    },
    {
      path := "../drake/examples/quadrotor/office.urdf"
      concept := "URDF office environment with fixed walls, table, cabinet, and drawers"
    },
    {
      path := "../drake/examples/quadrotor/quadrotor_geometry.h"
      concept := "declares the QuadrotorGeometry SceneGraph helper with state input, geometry_pose output, and body frame accessor"
    },
    {
      path := "../drake/examples/quadrotor/quadrotor_geometry.cc"
      concept := "parses the Skydio model into SceneGraph and maps position plus roll-pitch-yaw state to a FramePoseVector"
    },
    {
      path := "../drake/examples/quadrotor/test/quadrotor_geometry_test.cc"
      concept := "acceptance test for adding QuadrotorGeometry to a DiagramBuilder with QuadrotorPlant and SceneGraph"
    }
  ]

def stateCoordinateNames : Array String :=
  #["x", "y", "z", "roll", "pitch", "yaw",
    "xdot", "ydot", "zdot", "rolldot", "pitchdot", "yawdot"]

def inputCoordinateNames : Array String :=
  #["propeller_force_0", "propeller_force_1", "propeller_force_2", "propeller_force_3"]

def parameterCoordinateNames : Array String :=
  #["m", "L", "ixx", "iyy", "izz", "kF", "kM", "gravity"]

private def finitePositive (x : Float) : Bool :=
  Float.isFinite x && x > 0.0

private def finiteNonnegative (x : Float) : Bool :=
  Float.isFinite x && x >= 0.0

structure QuadrotorParams where
  mass : Float := 0.775
  armLength : Float := 0.15
  ixx : Float := 0.0015
  iyy : Float := 0.0025
  izz : Float := 0.0035
  forceConstant : Float := 1.0
  momentConstant : Float := 0.0245
  gravity : Float := 9.81
  stepSize : Float := 1.0e-3
  deriving Repr, Inhabited

namespace QuadrotorParams

def isValid (p : QuadrotorParams) : Bool :=
  finitePositive p.mass &&
  finiteNonnegative p.armLength &&
  finitePositive p.ixx &&
  finitePositive p.iyy &&
  finitePositive p.izz &&
  finiteNonnegative p.forceConstant &&
  finiteNonnegative p.momentConstant &&
  finiteNonnegative p.gravity

end QuadrotorParams

def params : QuadrotorParams := {}

structure Vec3 where
  x : Float := 0.0
  y : Float := 0.0
  z : Float := 0.0
  deriving Repr, BEq, Inhabited

namespace Vec3

def add (a b : Vec3) : Vec3 :=
  { x := a.x + b.x, y := a.y + b.y, z := a.z + b.z }

def sub (a b : Vec3) : Vec3 :=
  { x := a.x - b.x, y := a.y - b.y, z := a.z - b.z }

def scale (s : Float) (v : Vec3) : Vec3 :=
  { x := s * v.x, y := s * v.y, z := s * v.z }

def cross (a b : Vec3) : Vec3 :=
  {
    x := a.y * b.z - a.z * b.y
    y := a.z * b.x - a.x * b.z
    z := a.x * b.y - a.y * b.x
  }

def asArray (v : Vec3) : Array Float :=
  #[v.x, v.y, v.z]

def toSceneVec3 (v : Vec3) : SceneVec3 :=
  { x := v.x, y := v.y, z := v.z }

def isFinite (v : Vec3) : Bool :=
  Float.isFinite v.x && Float.isFinite v.y && Float.isFinite v.z

end Vec3

structure Rpy where
  roll : Float := 0.0
  pitch : Float := 0.0
  yaw : Float := 0.0
  deriving Repr, BEq, Inhabited

namespace Rpy

def isFinite (r : Rpy) : Bool :=
  Float.isFinite r.roll && Float.isFinite r.pitch && Float.isFinite r.yaw

def rotateBodyToWorld (r : Rpy) (v : Vec3) : Vec3 :=
  let sr := Float.sin r.roll
  let cr := Float.cos r.roll
  let sp := Float.sin r.pitch
  let cp := Float.cos r.pitch
  let sy := Float.sin r.yaw
  let cy := Float.cos r.yaw
  {
    x := cy * cp * v.x + (cy * sp * sr - sy * cr) * v.y +
      (cy * sp * cr + sy * sr) * v.z
    y := sy * cp * v.x + (sy * sp * sr + cy * cr) * v.y +
      (sy * sp * cr - cy * sr) * v.z
    z := -sp * v.x + cp * sr * v.y + cp * cr * v.z
  }

def bodyAngularVelocityFromRpyDt (r : Rpy) (rpyDt : Vec3) : Vec3 :=
  let sr := Float.sin r.roll
  let cr := Float.cos r.roll
  let sp := Float.sin r.pitch
  let cp := Float.cos r.pitch
  {
    x := rpyDt.x - rpyDt.z * sp
    y := rpyDt.y * cr + rpyDt.z * sr * cp
    z := -rpyDt.y * sr + rpyDt.z * cr * cp
  }

def bodyAngularVelocityBiasFromRpyDt (r : Rpy) (rpyDt : Vec3) : Vec3 :=
  let sr := Float.sin r.roll
  let cr := Float.cos r.roll
  let sp := Float.sin r.pitch
  let cp := Float.cos r.pitch
  let rd := rpyDt.x
  let pd := rpyDt.y
  let yd := rpyDt.z
  {
    x := -yd * pd * cp
    y := -sr * rd * pd + yd * cr * cp * rd - yd * sr * sp * pd
    z := -cr * rd * pd - yd * sr * cp * rd - yd * cr * sp * pd
  }

def rpyDtFromBodyAngularVelocity? (r : Rpy) (w : Vec3) : Except String Vec3 := do
  let sr := Float.sin r.roll
  let cr := Float.cos r.roll
  let sp := Float.sin r.pitch
  let cp := Float.cos r.pitch
  if Float.abs cp < 1.0e-12 then
    .error "quadrotor rpy pitch is too close to a kinematic singularity"
  else
    let tanPitch := sp / cp
    pure {
      x := w.x + sr * tanPitch * w.y + cr * tanPitch * w.z
      y := cr * w.y - sr * w.z
      z := (sr * w.y + cr * w.z) / cp
    }

def rpyDDtFromBodyAngularAcceleration?
    (r : Rpy) (rpyDt alphaBody : Vec3) : Except String Vec3 := do
  let bias := bodyAngularVelocityBiasFromRpyDt r rpyDt
  rpyDtFromBodyAngularVelocity? r (Vec3.sub alphaBody bias)

end Rpy

def defaultState : QuadrotorState :=
  Array.replicate 12 0.0

def mkState
    (x y z roll pitch yaw xdot ydot zdot rolldot pitchdot yawdot : Float) :
    QuadrotorState :=
  #[x, y, z, roll, pitch, yaw, xdot, ydot, zdot, rolldot, pitchdot, yawdot]

def defaultInput : QuadrotorInput :=
  Array.replicate 4 0.0

structure QuadrotorPhysicsState where
  state : QuadrotorState := defaultState
  input : QuadrotorInput := defaultInput
  deriving Repr, Inhabited

def physicsState
    (state : QuadrotorState := defaultState)
    (input : QuadrotorInput := defaultInput) : QuadrotorPhysicsState :=
  { state := state, input := input }

def hoverInput (p : QuadrotorParams := params) : QuadrotorInput :=
  Array.replicate 4 (p.mass * p.gravity / (4.0 * p.forceConstant))

def nominalHoverPosition : Vec3 :=
  { x := 0.0, y := 0.0, z := 1.0 }

def nominalHoverState : QuadrotorState :=
  mkState nominalHoverPosition.x nominalHoverPosition.y nominalHoverPosition.z
    0.0 0.0 0.0
    0.0 0.0 0.0
    0.0 0.0 0.0

def stateAsArray (x : QuadrotorState) : Array Float :=
  x

def inputAsArray (u : QuadrotorInput) : Array Float :=
  u

def stateIsValid (x : QuadrotorState) : Bool :=
  x.size == 12 && x.all Float.isFinite

def inputIsValid (u : QuadrotorInput) : Bool :=
  u.size == 4 && u.all Float.isFinite

def position (x : QuadrotorState) : Vec3 :=
  { x := x.getD 0 0.0, y := x.getD 1 0.0, z := x.getD 2 0.0 }

def rpy (x : QuadrotorState) : Rpy :=
  { roll := x.getD 3 0.0, pitch := x.getD 4 0.0, yaw := x.getD 5 0.0 }

def translationalVelocity (x : QuadrotorState) : Vec3 :=
  { x := x.getD 6 0.0, y := x.getD 7 0.0, z := x.getD 8 0.0 }

def rpyDt (x : QuadrotorState) : Vec3 :=
  { x := x.getD 9 0.0, y := x.getD 10 0.0, z := x.getD 11 0.0 }

/-! ## QuadrotorGeometry SceneGraph provider -/

def quadrotorModelUri : String :=
  "package://drake_models/skydio_2/quadrotor.urdf"

def quadrotorGeometrySourceId : Nat := 5360
def quadrotorBodyFrameId : Nat := 5361
def quadrotorBodyGeometryId : Nat := 5362

def quadrotorGeometryStateInputVertex : VertexId := 5363
def quadrotorGeometryProviderVertex : VertexId := 5364
def quadrotorGeometryPoseOutputVertex : VertexId := 5365

private def quadrotorIllustrationProperties : SceneGeometryProperties :=
  {
    roles := #[.illustration]
    diffuseRgba? := some { r := 0.78, g := 0.78, b := 0.82, a := 1.0 }
  }

def quadrotorGeometryProvider : SceneGraphProvider :=
  {
    sources := #[
      { id := quadrotorGeometrySourceId, name := "QuadrotorGeometry" }
    ]
    frames := #[
      {
        id := quadrotorBodyFrameId
        sourceId := quadrotorGeometrySourceId
        name := "base_link"
      }
    ]
    geometries := #[
      {
        id := quadrotorBodyGeometryId
        sourceId := quadrotorGeometrySourceId
        frameId? := some quadrotorBodyFrameId
        X_FG := ScenePose3.identity
        shape := .model quadrotorModelUri
        name := "skydio_2_quadrotor_model"
        properties := quadrotorIllustrationProperties
      }
    ]
    label := "QuadrotorGeometry SceneGraph provider"
  }

def quadrotorGeometryPoseOutput
    (x : QuadrotorState := nominalHoverState) : SceneFramePoseVector :=
  let p := position x
  let attitude := rpy x
  {
    poses := #[
      {
        frameId := quadrotorBodyFrameId
        X_WF := ScenePose3.fromRollPitchYaw
          { x := p.x, y := p.y, z := p.z }
          attitude.roll attitude.pitch attitude.yaw
      }
    ]
  }

/-! ## Environment model assets and contact candidate provider -/

inductive QuadrotorEnvironmentFormat where
  | urdf
  | sdf
  deriving Repr, BEq, Inhabited

namespace QuadrotorEnvironmentFormat

def matchesPath : QuadrotorEnvironmentFormat → String → Bool
  | .urdf, path => path.endsWith ".urdf"
  | .sdf, path => path.endsWith ".sdf"

end QuadrotorEnvironmentFormat

structure QuadrotorObstacleBox where
  name : String
  center : SceneVec3
  size : SceneVec3
  material? : Option String := none
  hasVisualGeometry : Bool := true
  hasCollisionGeometry : Bool := true
  deriving Repr, BEq, Inhabited

namespace QuadrotorObstacleBox

def validate? (box : QuadrotorObstacleBox) : Except String Unit := do
  if box.name.isEmpty then
    .error "quadrotor obstacle box name cannot be empty"
  if !box.center.isFinite then
    .error s!"quadrotor obstacle {box.name}: center must be finite"
  if !box.size.isFinite || box.size.x <= 0.0 || box.size.y <= 0.0 || box.size.z <= 0.0 then
    .error s!"quadrotor obstacle {box.name}: size must be positive and finite"
  if !box.hasVisualGeometry && !box.hasCollisionGeometry then
    .error s!"quadrotor obstacle {box.name}: visual or collision geometry must be present"

def halfExtent (box : QuadrotorObstacleBox) : SceneVec3 :=
  { x := 0.5 * box.size.x, y := 0.5 * box.size.y, z := 0.5 * box.size.z }

private def signNonzero (x : Float) : Float :=
  if x < 0.0 then -1.0 else 1.0

private def axisNormal (axis : Nat) (sign : Float) : SceneVec3 :=
  if axis == 0 then { x := sign }
  else if axis == 1 then { y := sign }
  else { z := sign }

private def normalizeOr (fallback : SceneVec3) (v : SceneVec3) : SceneVec3 :=
  match v.normalize? with
  | .ok n => n
  | .error _ => fallback

private def minInsideAxis
    (clearX clearY clearZ : Float) : Nat :=
  if clearX <= clearY && clearX <= clearZ then 0
  else if clearY <= clearZ then 1
  else 2

def signedDistanceAndNormal (box : QuadrotorObstacleBox) (p : SceneVec3) :
    Float × SceneVec3 :=
  let h := box.halfExtent
  let rel := p.sub box.center
  let ax := Float.abs rel.x
  let ay := Float.abs rel.y
  let az := Float.abs rel.z
  let dx := ax - h.x
  let dy := ay - h.y
  let dz := az - h.z
  let ox := max dx 0.0
  let oy := max dy 0.0
  let oz := max dz 0.0
  if ox > 0.0 || oy > 0.0 || oz > 0.0 then
    let dist := Float.sqrt (ox * ox + oy * oy + oz * oz)
    let normal :=
      if dx > 0.0 && dy <= 0.0 && dz <= 0.0 then
        axisNormal 0 (signNonzero rel.x)
      else if dy > 0.0 && dx <= 0.0 && dz <= 0.0 then
        axisNormal 1 (signNonzero rel.y)
      else if dz > 0.0 && dx <= 0.0 && dy <= 0.0 then
        axisNormal 2 (signNonzero rel.z)
      else
        normalizeOr SceneVec3.unitZ
          { x := if dx > 0.0 then signNonzero rel.x * ox else 0.0
            y := if dy > 0.0 then signNonzero rel.y * oy else 0.0
            z := if dz > 0.0 then signNonzero rel.z * oz else 0.0 }
    (dist, normal)
  else
    let clearX := h.x - ax
    let clearY := h.y - ay
    let clearZ := h.z - az
    let axis := minInsideAxis clearX clearY clearZ
    let penetration :=
      if axis == 0 then clearX else if axis == 1 then clearY else clearZ
    let sign :=
      if axis == 0 then signNonzero rel.x
      else if axis == 1 then signNonzero rel.y
      else signNonzero rel.z
    (-penetration, axisNormal axis sign)

end QuadrotorObstacleBox

structure QuadrotorEnvironmentAsset where
  path : String
  packageUri : String
  format : QuadrotorEnvironmentFormat
  modelName : String
  staticModel : Bool := true
  linkNames : Array String := #[]
  jointNames : Array String := #[]
  materialNames : Array String := #[]
  obstacleBoxes : Array QuadrotorObstacleBox := #[]
  loadedBy : Array String := #[]
  deriving Repr, Inhabited

namespace QuadrotorEnvironmentAsset

def validate? (asset : QuadrotorEnvironmentAsset) : Except String Unit := do
  if asset.path.isEmpty then
    .error "quadrotor environment asset path cannot be empty"
  if asset.packageUri.isEmpty then
    .error s!"quadrotor environment asset {asset.path}: package URI cannot be empty"
  if !asset.format.matchesPath asset.path then
    .error s!"quadrotor environment asset {asset.path}: format does not match path"
  if asset.modelName.isEmpty then
    .error s!"quadrotor environment asset {asset.path}: model name cannot be empty"
  if asset.linkNames.isEmpty then
    .error s!"quadrotor environment asset {asset.path}: link names cannot be empty"
  if asset.obstacleBoxes.isEmpty then
    .error s!"quadrotor environment asset {asset.path}: obstacle boxes cannot be empty"
  for box in asset.obstacleBoxes do
    box.validate?

end QuadrotorEnvironmentAsset

private def box (name : String) (center size : SceneVec3)
    (material? : Option String := none) : QuadrotorObstacleBox :=
  { name := name, center := center, size := size, material? := material? }

def warehouseObstacleBoxes : Array QuadrotorObstacleBox :=
  #[
    box "bottom_wall" { x := 3.5, y := -0.25, z := -1.6 } { x := 10.0, y := 4.5, z := 0.2 },
    box "left_wall" { x := 3.5, y := 1.75, z := 0.0 } { x := 9.0, y := 0.5, z := 3.0 },
    box "right_wall" { x := 3.5, y := -2.25, z := 0.0 } { x := 9.0, y := 0.5, z := 3.0 },
    box "back_wall" { x := -1.25, y := -0.25, z := 0.0 } { x := 0.5, y := 4.5, z := 3.0 },
    box "front_wall" { x := 8.25, y := -0.25, z := 0.0 } { x := 0.5, y := 4.5, z := 3.0 },
    box "slalom_1" { x := 1.75, y := 0.5, z := 0.0 } { x := 0.5, y := 2.0, z := 3.0 },
    box "slalom_2" { x := 3.75, y := -1.0, z := 0.0 } { x := 0.5, y := 2.0, z := 3.0 },
    box "slalom_3" { x := 5.75, y := 0.5, z := 0.0 } { x := 0.5, y := 2.0, z := 3.0 }
  ]

def officeObstacleBoxes : Array QuadrotorObstacleBox :=
  #[
    box "wall1" { z := 1.0 } { x := 10.0, y := 0.1, z := 2.0 } (some "Brown"),
    box "wall2" {} { x := 0.1, y := 8.0, z := 2.0 } (some "Brown"),
    box "wall3" { z := 0.5 } { x := 10.0, y := 0.1, z := 1.0 } (some "Brown"),
    box "wall4" {} { x := 0.1, y := 8.0, z := 2.0 } (some "Brown"),
    box "internalWall" {} { x := 7.5, y := 0.1, z := 2.0 } (some "Brown"),
    box "tableLegUL" { x := -1.0, y := 3.0, z := 0.5 } { x := 0.1, y := 0.1, z := 1.0 } (some "White"),
    box "wall23" { x := 1.75, z := 1.0 } { x := 6.5, y := 0.1, z := 2.0 } (some "Brown"),
    box "wall34" { x := -4.125, z := 1.0 } { x := 1.75, y := 0.1, z := 2.0 } (some "Brown"),
    box "tableLegUR" { x := 1.0, y := 3.0, z := 0.5 } { x := 0.1, y := 0.1, z := 1.0 } (some "White"),
    box "tableLegLR" { x := 1.0, y := 1.0, z := 0.5 } { x := 0.1, y := 0.1, z := 1.0 } (some "White"),
    box "tableLegLL" { x := -1.0, y := 1.0, z := 0.5 } { x := 0.1, y := 0.1, z := 1.0 } (some "White"),
    box "TableTop" { y := 2.0, z := 0.98 } { x := 2.1, y := 2.1, z := 0.1 } (some "Grey"),
    box "cabinet" { x := -3.3, y := 5.5, z := 0.9 } { x := 2.2, y := 1.0, z := 1.8 } (some "Grey"),
    box "drawer1" { x := -3.3, y := 5.3, z := 0.9 } { x := 2.0, y := 1.0, z := 0.33 } (some "White"),
    box "drawer2" { x := -3.3, y := 5.3, z := 1.4 } { x := 2.0, y := 1.0, z := 0.33 } (some "White"),
    box "drawer3" { x := -3.3, y := 5.3, z := 0.4 } { x := 2.0, y := 1.0, z := 0.33 } (some "White")
  ]

def warehouseEnvironmentAsset : QuadrotorEnvironmentAsset :=
  {
    path := "../drake/examples/quadrotor/warehouse.sdf"
    packageUri := "package://drake/examples/quadrotor/warehouse.sdf"
    format := .sdf
    modelName := "room_w_lidar"
    staticModel := true
    linkNames := warehouseObstacleBoxes.map (fun box => box.name)
    obstacleBoxes := warehouseObstacleBoxes
    loadedBy := #["../drake/examples/quadrotor/run_quadrotor_dynamics.cc"]
  }

def officeEnvironmentAsset : QuadrotorEnvironmentAsset :=
  {
    path := "../drake/examples/quadrotor/office.urdf"
    packageUri := "package://drake/examples/quadrotor/office.urdf"
    format := .urdf
    modelName := "office"
    staticModel := true
    linkNames := officeObstacleBoxes.map (fun box => box.name)
    jointNames := #[
      "joint1", "joint2", "joint3", "lowerToUpperWall3From2",
      "lowerToUpperWall3From4", "internalJoint", "WallToTableTop",
      "TableTopToLegUL", "TableTopToLegUR", "TableTopToLegLR",
      "TableTopToLegLL", "WallToCabinet", "CabinetToDrawer1",
      "CabinetToDrawer2", "CabinetToDrawer3"
    ]
    materialNames := #["Brown", "White", "Grey", "Red"]
    obstacleBoxes := officeObstacleBoxes
  }

def quadrotorEnvironmentAssets : Array QuadrotorEnvironmentAsset :=
  #[warehouseEnvironmentAsset, officeEnvironmentAsset]

def findQuadrotorEnvironmentAsset? (path : String) :
    Option QuadrotorEnvironmentAsset :=
  quadrotorEnvironmentAssets.find? (fun asset =>
    asset.path == path || asset.packageUri == path)

def validateQuadrotorEnvironmentAssets? : Except String Unit := do
  if quadrotorEnvironmentAssets.size != 2 then
    .error s!"quadrotor environment catalog size {quadrotorEnvironmentAssets.size} != 2"
  for asset in quadrotorEnvironmentAssets do
    asset.validate?
  if (findQuadrotorEnvironmentAsset? "../drake/examples/quadrotor/warehouse.sdf").isNone then
    .error "quadrotor environment catalog missing warehouse.sdf"
  if (findQuadrotorEnvironmentAsset? "../drake/examples/quadrotor/office.urdf").isNone then
    .error "quadrotor environment catalog missing office.urdf"

def quadrotorEnvironmentSourceId : Nat := 5380
def quadrotorEnvironmentGeometryIdBase : Nat := 5381

private def colorForMaterial? : Option String → Option SceneRgba
  | some "Brown" => some { r := 0.33, g := 0.21, b := 0.04, a := 1.0 }
  | some "White" => some { r := 1.0, g := 1.0, b := 1.0, a := 1.0 }
  | some "Grey" => some { r := 0.3, g := 0.3, b := 0.3, a := 1.0 }
  | some "Red" => some { r := 1.0, g := 0.0, b := 0.0, a := 1.0 }
  | _ => some { r := 0.55, g := 0.58, b := 0.62, a := 1.0 }

def environmentSceneGraphProvider
    (asset : QuadrotorEnvironmentAsset := warehouseEnvironmentAsset) :
    SceneGraphProvider := Id.run do
  let mut geometries : Array SceneGeometry := #[]
  for i in [:asset.obstacleBoxes.size] do
    let obstacle := asset.obstacleBoxes[i]!
    geometries := geometries.push {
      id := quadrotorEnvironmentGeometryIdBase + i
      sourceId := quadrotorEnvironmentSourceId
      frameId? := none
      X_FG := ScenePose3.translated obstacle.center
      shape := .box obstacle.size.x obstacle.size.y obstacle.size.z
      name := obstacle.name
      properties := {
        roles := #[.illustration, .proximity]
        diffuseRgba? := colorForMaterial? obstacle.material?
        friction := { staticFriction := 0.7, dynamicFriction := 0.5 }
      }
    }
  return {
    sources := #[{ id := quadrotorEnvironmentSourceId, name := asset.modelName }]
    geometries := geometries
    label := s!"quadrotor environment provider:{asset.modelName}"
  }

private def sceneRow6 (v : SceneVec3) : Array Float :=
  #[v.x, v.y, v.z, 0.0, 0.0, 0.0]

private def tangent1ForNormal (n : SceneVec3) : SceneVec3 :=
  if Float.abs n.z < 0.9 then
    match (SceneVec3.unitZ.cross n).normalize? with
    | .ok t => t
    | .error _ => SceneVec3.unitX
  else
    SceneVec3.unitX

private def tangent2ForNormal (n : SceneVec3) : SceneVec3 :=
  let t1 := tangent1ForNormal n
  match (n.cross t1).normalize? with
  | .ok t => t
  | .error _ => SceneVec3.unitY

def warehouseContactCandidateProvider :
    ContactCandidateProvider QuadrotorState :=
  {
    label := "quadrotor warehouse static obstacle provider"
    candidatesAt? := fun x =>
      if !stateIsValid x then
        .error "quadrotor warehouse contact provider: invalid 12-state vector"
      else
        let p := (position x).toSceneVec3
        let v := (translationalVelocity x).toSceneVec3
        let candidates : Array ContactCandidate := Id.run do
          let mut out : Array ContactCandidate := #[]
          for i in [:warehouseObstacleBoxes.size] do
            let obstacle := warehouseObstacleBoxes[i]!
            let (distance, normal) := obstacle.signedDistanceAndNormal p
            let t1 := tangent1ForNormal normal
            let t2 := tangent2ForNormal normal
            out := out.push {
              id := quadrotorEnvironmentGeometryIdBase + i
              bodyA := "quadrotor"
              bodyB := obstacle.name
              point_W := p.toArray
              normal_W := normal.toArray
              signedDistance := distance
              normalVelocity := normal.dot v
              tangentVelocity := t1.dot v
              tangentVelocity2 := t2.dot v
              normalJacobian := sceneRow6 normal
              tangentJacobian := sceneRow6 t1
              tangentJacobian2 := sceneRow6 t2
              label := s!"quadrotor warehouse contact:{obstacle.name}"
            }
          return out
        .ok {
          candidates := candidates
          sourceCandidateCount? := some warehouseObstacleBoxes.size
          label := "quadrotor warehouse contact candidates"
        }
  }

private def quadrotorGeometryMove
    (target : VertexId) (label : String) (reads : Array VertexId := #[])
    (writes : Array VertexId := #[]) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[target]
    reads := reads
    writes := writes
    exactness := .exact
    label := label
  }

def quadrotorGeometryGraph : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex {
      id := quadrotorGeometryStateInputVertex
      kind := .state .boundary
      label := "QuadrotorGeometry state input"
    }
    |>.addVertex {
      id := quadrotorGeometryProviderVertex
      kind := .state .boundary
      label := "QuadrotorGeometry registered SceneGraph source"
    }
    |>.addVertex {
      id := quadrotorGeometryPoseOutputVertex
      kind := .state .checkpoint
      label := "QuadrotorGeometry geometry_pose output"
    }
    |>.addMove (quadrotorGeometryMove quadrotorGeometryProviderVertex
      "Parse Skydio quadrotor URDF into SceneGraph source/frame/model geometry"
      #[] #[quadrotorGeometryProviderVertex])
    |>.addMove (quadrotorGeometryMove quadrotorGeometryPoseOutputVertex
      "OutputGeometryPose: x,rpy -> body FramePoseVector"
      #[quadrotorGeometryStateInputVertex, quadrotorGeometryProviderVertex]
      #[quadrotorGeometryPoseOutputVertex])

structure QuadrotorGeometryResult where
  references : Array DrakeReference
  modelUri : String
  inputPortName : String := "state"
  inputPortSize : Nat := 12
  outputPortName : String := "geometry_pose"
  provider : SceneGraphProvider
  sampleState : QuadrotorState
  poses : SceneFramePoseVector
  graph : SkeletonGraph
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def buildQuadrotorGeometry?
    (x : QuadrotorState := nominalHoverState) : Except String QuadrotorGeometryResult := do
  if !stateIsValid x then
    .error "QuadrotorGeometry state input must have twelve finite coordinates"
  quadrotorGeometryProvider.validate?
  let poses := quadrotorGeometryPoseOutput x
  poses.validate? quadrotorGeometryProvider
  pure {
    references := drakeReferences
    modelUri := quadrotorModelUri
    provider := quadrotorGeometryProvider
    sampleState := x
    poses := poses
    graph := quadrotorGeometryGraph
    moves := quadrotorGeometryGraph.moves
  }

def rotorBodyForces (p : QuadrotorParams) (u : QuadrotorInput) : Array Float :=
  u.map (fun ui => p.forceConstant * ui)

def bodyThrust (p : QuadrotorParams) (u : QuadrotorInput) : Vec3 :=
  { z := (rotorBodyForces p u).foldl (fun acc f => acc + f) 0.0 }

def bodyTorque (p : QuadrotorParams) (u : QuadrotorInput) : Vec3 :=
  let f := rotorBodyForces p u
  {
    x := p.armLength * (f.getD 1 0.0 - f.getD 3 0.0)
    y := p.armLength * (f.getD 2 0.0 - f.getD 0 0.0)
    z := p.momentConstant *
      (u.getD 0 0.0 - u.getD 1 0.0 + u.getD 2 0.0 - u.getD 3 0.0)
  }

def allocateRotorInput
    (p : QuadrotorParams)
    (totalThrust : Float)
    (tau : Vec3) : QuadrotorInput :=
  let mx := tau.x / p.armLength
  let my := tau.y / p.armLength
  let mz := tau.z / p.momentConstant
  let evenPair := 0.5 * (totalThrust + mz)
  let oddPair := 0.5 * (totalThrust - mz)
  #[
    0.5 * (evenPair - my),
    0.5 * (oddPair + mx),
    0.5 * (evenPair + my),
    0.5 * (oddPair - mx)
  ]

def translationalAcceleration
    (p : QuadrotorParams) (u : QuadrotorInput) (x : QuadrotorState) : Vec3 :=
  let thrustWorld := (rpy x).rotateBodyToWorld (bodyThrust p u)
  Vec3.add { z := -p.gravity } (Vec3.scale (1.0 / p.mass) thrustWorld)

def bodyAngularVelocity (x : QuadrotorState) : Vec3 :=
  (rpy x).bodyAngularVelocityFromRpyDt (rpyDt x)

def inertiaTimes (p : QuadrotorParams) (w : Vec3) : Vec3 :=
  { x := p.ixx * w.x, y := p.iyy * w.y, z := p.izz * w.z }

def bodyAngularAcceleration
    (p : QuadrotorParams) (u : QuadrotorInput) (x : QuadrotorState) : Vec3 :=
  let w := bodyAngularVelocity x
  let gyroscopic := Vec3.cross w (inertiaTimes p w)
  let rhs := Vec3.sub (bodyTorque p u) gyroscopic
  { x := rhs.x / p.ixx, y := rhs.y / p.iyy, z := rhs.z / p.izz }

def quadrotorFullPhysicsIntervalVertex : VertexId := 5310

def massMatrix (p : QuadrotorParams) : Array (Array Float) :=
  FloatMatrix.diagonal #[p.mass, p.mass, p.mass, p.ixx, p.iyy, p.izz]

def poseQdot (x : QuadrotorState) : Array Float :=
  let v := translationalVelocity x
  let rdot := rpyDt x
  #[v.x, v.y, v.z, rdot.x, rdot.y, rdot.z]

def gravityGeneralizedForce (p : QuadrotorParams) : Array Float :=
  #[0.0, 0.0, -p.mass * p.gravity, 0.0, 0.0, 0.0]

def rotorGeneralizedForce
    (p : QuadrotorParams) (u : QuadrotorInput) (x : QuadrotorState) :
    Array Float :=
  let thrustWorld := (rpy x).rotateBodyToWorld (bodyThrust p u)
  let tau := bodyTorque p u
  #[thrustWorld.x, thrustWorld.y, thrustWorld.z, tau.x, tau.y, tau.z]

def gyroscopicBiasForce (p : QuadrotorParams) (x : QuadrotorState) :
    Array Float :=
  let w := bodyAngularVelocity x
  let gyro := Vec3.cross w (inertiaTimes p w)
  #[0.0, 0.0, 0.0, gyro.x, gyro.y, gyro.z]

def validateFullPhysicsInputs?
    (p : QuadrotorParams) (u : QuadrotorInput) (x : QuadrotorState) :
    Except String Unit := do
  if !p.isValid then
    .error "quadrotor params are invalid"
  if !inputIsValid u then
    .error "quadrotor input must have four finite propeller force coordinates"
  if !stateIsValid x then
    .error "quadrotor state must have twelve finite coordinates"

def fullPhysicsPrimitives
    (p : QuadrotorParams) (u : QuadrotorInput) (x : QuadrotorState)
    (label : String := "quadrotor full physics") : FullPhysicsPrimitives :=
  {
    massMatrix := massMatrix p
    qdot := poseQdot x
    actuationForces := rotorGeneralizedForce p u x
    biasForces := gyroscopicBiasForce p x
    generalizedForceContributions :=
      #[GeneralizedForceContribution.ofForce
          (gravityGeneralizedForce p)
          "quadrotor gravity generalized force"
          "Quadrotor"]
    label := label
  }

def fullPhysicsPrimitiveProvider
    (p : QuadrotorParams := params)
    (label : String := "quadrotor full physics provider") :
    FullPhysicsPrimitiveProvider QuadrotorPhysicsState :=
  {
    label := label
    primitivesAt? := fun snapshot => do
      validateFullPhysicsInputs? p snapshot.input snapshot.state
      pure (fullPhysicsPrimitives p snapshot.input snapshot.state label)
  }

def solveFullPhysics?
    (p : QuadrotorParams) (u : QuadrotorInput) (x : QuadrotorState)
    (label : String := "quadrotor full physics") :
    Except String FullPhysicsResult := do
  validateFullPhysicsInputs? p u x
  let equation ← (fullPhysicsPrimitives p u x label).equation?
  equation.solve? quadrotorFullPhysicsIntervalVertex

def derivative? (p : QuadrotorParams) (u : QuadrotorInput) (x : QuadrotorState) :
    Except String QuadrotorState := do
  let fullPhysics ← solveFullPhysics? p u x
  let vdot := fullPhysics.derivative.vdot
  let a : Vec3 := {
    x := vdot.getD 0 0.0
    y := vdot.getD 1 0.0
    z := vdot.getD 2 0.0
  }
  let alphaBody : Vec3 := {
    x := vdot.getD 3 0.0
    y := vdot.getD 4 0.0
    z := vdot.getD 5 0.0
  }
  let rpyDDt ← (rpy x).rpyDDtFromBodyAngularAcceleration? (rpyDt x) alphaBody
  let v := translationalVelocity x
  let rdot := rpyDt x
  pure #[v.x, v.y, v.z, rdot.x, rdot.y, rdot.z,
    a.x, a.y, a.z, rpyDDt.x, rpyDDt.y, rpyDDt.z]

def derivative (p : QuadrotorParams) (u : QuadrotorInput := defaultInput)
    (x : QuadrotorState) : QuadrotorState :=
  match derivative? p u x with
  | .ok dx => dx
  | .error _ => defaultState

def kineticEnergy (p : QuadrotorParams) (x : QuadrotorState) : Float :=
  let v := translationalVelocity x
  let w := bodyAngularVelocity x
  0.5 * p.mass * (v.x * v.x + v.y * v.y + v.z * v.z) +
    0.5 * (p.ixx * w.x * w.x + p.iyy * w.y * w.y + p.izz * w.z * w.z)

def potentialEnergy (p : QuadrotorParams) (x : QuadrotorState) : Float :=
  p.mass * p.gravity * (position x).z

def totalEnergy (p : QuadrotorParams) (x : QuadrotorState) : Float :=
  kineticEnergy p x + potentialEnergy p x

private def clamp (lo hi x : Float) : Float :=
  min hi (max lo x)

structure LqrConfig where
  nominalPosition : Vec3 := nominalHoverPosition
  qPose : Float := 10.0
  qVelocity : Float := 1.0
  rInput : Float := 1.0
  duration : Float := 7.0
  stepSize : Float := 1.0e-3
  steps : Nat := 7000
  kpXY : Float := 1.8
  kdXY : Float := 2.4
  kpZ : Float := 5.0
  kdZ : Float := 4.0
  kpRollPitch : Float := 18.0
  kdRollPitch : Float := 8.0
  kpYaw : Float := 6.0
  kdYaw : Float := 4.0
  maxTilt : Float := 0.45
  deriving Repr, Inhabited

namespace LqrConfig

def nominalState (cfg : LqrConfig) : QuadrotorState :=
  mkState cfg.nominalPosition.x cfg.nominalPosition.y cfg.nominalPosition.z
    0.0 0.0 0.0
    0.0 0.0 0.0
    0.0 0.0 0.0

def validate? (cfg : LqrConfig) : Except String Unit := do
  if !cfg.nominalPosition.isFinite then
    .error "quadrotor LQR nominal position must be finite"
  if !(Float.isFinite cfg.qPose) || cfg.qPose < 0.0 then
    .error s!"quadrotor LQR qPose must be nonnegative and finite, got {cfg.qPose}"
  if !(Float.isFinite cfg.qVelocity) || cfg.qVelocity < 0.0 then
    .error s!"quadrotor LQR qVelocity must be nonnegative and finite, got {cfg.qVelocity}"
  if !(Float.isFinite cfg.rInput) || cfg.rInput <= 0.0 then
    .error s!"quadrotor LQR rInput must be positive and finite, got {cfg.rInput}"
  if !(Float.isFinite cfg.duration) || cfg.duration <= 0.0 then
    .error s!"quadrotor LQR duration must be positive and finite, got {cfg.duration}"
  if !(Float.isFinite cfg.stepSize) || cfg.stepSize <= 0.0 then
    .error s!"quadrotor LQR step size must be positive and finite, got {cfg.stepSize}"
  if cfg.steps == 0 then
    .error "quadrotor LQR rollout requires at least one step"
  if Float.abs (cfg.steps.toFloat * cfg.stepSize - cfg.duration) > 1.0e-12 then
    .error s!"quadrotor LQR step count {cfg.steps} does not match duration {cfg.duration}"
  for (name, value) in #[
      ("kpXY", cfg.kpXY), ("kdXY", cfg.kdXY), ("kpZ", cfg.kpZ), ("kdZ", cfg.kdZ),
      ("kpRollPitch", cfg.kpRollPitch), ("kdRollPitch", cfg.kdRollPitch),
      ("kpYaw", cfg.kpYaw), ("kdYaw", cfg.kdYaw), ("maxTilt", cfg.maxTilt)] do
    if !(Float.isFinite value) || value < 0.0 then
      .error s!"quadrotor LQR gain {name} must be nonnegative and finite, got {value}"

end LqrConfig

def lqrConfig : LqrConfig := {}

structure LqrCostMetadata where
  qDiagonal : Array Float
  rDiagonal : Array Float
  nominalState : QuadrotorState
  nominalInput : QuadrotorInput
  deriving Repr, Inhabited

def lqrCostMetadata (p : QuadrotorParams := params) (cfg : LqrConfig := lqrConfig) :
    LqrCostMetadata :=
  {
    qDiagonal := Array.replicate 6 cfg.qPose ++ Array.replicate 6 cfg.qVelocity
    rDiagonal := Array.replicate 4 cfg.rInput
    nominalState := cfg.nominalState
    nominalInput := hoverInput p
  }

def lqrController
    (p : QuadrotorParams := params)
    (cfg : LqrConfig := lqrConfig)
    (x : QuadrotorState) : QuadrotorInput :=
  let pos := position x
  let vel := translationalVelocity x
  let attitude := rpy x
  let attitudeDot := rpyDt x
  let axCmd := cfg.kpXY * (cfg.nominalPosition.x - pos.x) - cfg.kdXY * vel.x
  let ayCmd := cfg.kpXY * (cfg.nominalPosition.y - pos.y) - cfg.kdXY * vel.y
  let azCmd := cfg.kpZ * (cfg.nominalPosition.z - pos.z) - cfg.kdZ * vel.z
  let rollDes := clamp (-cfg.maxTilt) cfg.maxTilt (-ayCmd / p.gravity)
  let pitchDes := clamp (-cfg.maxTilt) cfg.maxTilt (axCmd / p.gravity)
  let total := p.mass * (p.gravity + azCmd) /
    max 0.25 ((Float.cos attitude.roll) * (Float.cos attitude.pitch))
  let tau : Vec3 := {
    x := p.ixx * (cfg.kpRollPitch * (rollDes - attitude.roll) -
      cfg.kdRollPitch * attitudeDot.x)
    y := p.iyy * (cfg.kpRollPitch * (pitchDes - attitude.pitch) -
      cfg.kdRollPitch * attitudeDot.y)
    z := p.izz * (cfg.kpYaw * (-attitude.yaw) - cfg.kdYaw * attitudeDot.z)
  }
  allocateRotorInput p total tau

def odeTerm (p : QuadrotorParams) : ODETerm QuadrotorState QuadrotorInput :=
  { vectorField := fun _t x u => derivative p u x }

def quadrotorSolver :=
  RK4.solver
    (Term := ODETerm QuadrotorState QuadrotorInput)
    (Y := QuadrotorState)
    (VF := QuadrotorState)
    (Args := QuadrotorInput)

private def stateAddScaled (x dx : QuadrotorState) (h : Float) : QuadrotorState := Id.run do
  let mut out := #[]
  for i in [:12] do
    out := out.push (x.getD i 0.0 + h * dx.getD i 0.0)
  return out

private def rk4ClosedLoopStep
    (p : QuadrotorParams)
    (controller : QuadrotorState → QuadrotorInput)
    (dt : Float)
    (x : QuadrotorState) : QuadrotorState :=
  let f := fun y => derivative p (controller y) y
  let k1 := f x
  let k2 := f (stateAddScaled x k1 (0.5 * dt))
  let k3 := f (stateAddScaled x k2 (0.5 * dt))
  let k4 := f (stateAddScaled x k3 dt)
  Id.run do
    let mut out := #[]
    for i in [:12] do
      out := out.push
        (x.getD i 0.0 + dt *
          (k1.getD i 0.0 + 2.0 * k2.getD i 0.0 + 2.0 * k3.getD i 0.0 + k4.getD i 0.0) /
            6.0)
    return out

private def rolloutClosedLoop
    (p : QuadrotorParams)
    (controller : QuadrotorState → QuadrotorInput)
    (dt : Float)
    (steps : Nat)
    (x0 : QuadrotorState) : Array QuadrotorState := Id.run do
  let mut x := x0
  let mut samples := #[x0]
  for _ in [:steps] do
    x := rk4ClosedLoopStep p controller dt x
    samples := samples.push x
  return samples

structure SimulationResult where
  references : Array DrakeReference
  t0 : Float
  t1 : Float
  input : QuadrotorInput
  initialState : QuadrotorState
  finalState : QuadrotorState
  initialEnergy : Float
  finalEnergy : Float
  fullPhysics : FullPhysicsResult
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

def acceptedSegment (t0 t1 : Float) : AcceptedStepSegment :=
  {
    id := 0
    attemptIndex := 0
    tStart := t0
    tAttempt := t1
    tAfter := t1
    label := "quadrotor-continuous-interval"
  }

structure LqrSimulationResult where
  references : Array DrakeReference
  controllerName : String
  t0 : Float
  t1 : Float
  stepSize : Float
  cost : LqrCostMetadata
  initialState : QuadrotorState
  finalState : QuadrotorState
  samples : Array QuadrotorState
  finalInput : QuadrotorInput
  initialEnergy : Float
  finalEnergy : Float
  finalFullPhysics : FullPhysicsResult
  trace : DynamicEventTrace
  moves : Array SkeletonMove
  deriving Repr, Inhabited

private def controllerMove (vertex : VertexId) (label : String) : SkeletonMove :=
  {
    kind := .localSchurBlock
    targets := #[vertex]
    exactness := .controlledApproximation
    label := label
  }

def solve? (p : QuadrotorParams := params)
    (x0 : QuadrotorState := mkState 0.0 0.0 0.051 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)
    (t0 : Float := 0.0)
    (t1 : Float := 0.1)
    (u : QuadrotorInput := defaultInput) :
    Except String SimulationResult := do
  if !p.isValid then
    .error "quadrotor params are invalid"
  if !inputIsValid u then
    .error "quadrotor input is invalid"
  if !stateIsValid x0 then
    .error "quadrotor initial state is invalid"
  let sol :=
    diffeqsolve
      (Term := ODETerm QuadrotorState QuadrotorInput)
      (Y := QuadrotorState)
      (VF := QuadrotorState)
      (Control := Time)
      (Args := QuadrotorInput)
      (Controller := ConstantStepSize)
      (odeTerm p) quadrotorSolver t0 t1 (some p.stepSize) x0 u
      (saveat := { t1 := true })
  if !sol.result.isOkay then
    .error s!"quadrotor solve failed: {reprStr sol.result}"
  else
    match sol.ts, sol.ys with
    | some ts, some ys =>
        if ts.size == 0 || ys.size == 0 then
          .error "quadrotor solve did not save endpoint"
        else
          let final := ys[ys.size - 1]!
          let fullPhysics ← solveFullPhysics? p u x0 "quadrotor initial full physics"
          let trace := DynamicEventTrace.empty.push (.interval (acceptedSegment t0 ts[ts.size - 1]!))
          trace.validate?
          pure {
            references := drakeReferences
            t0 := t0
            t1 := ts[ts.size - 1]!
            input := u
            initialState := x0
            finalState := final
            initialEnergy := totalEnergy p x0
            finalEnergy := totalEnergy p final
            fullPhysics := fullPhysics
            trace := trace
            moves := trace.moves ++ #[fullPhysics.supportMove, fullPhysics.move]
          }
    | _, _ => .error "quadrotor solve did not save endpoint arrays"

def solvePassive? (p : QuadrotorParams := params)
    (x0 : QuadrotorState := mkState 0.0 0.0 0.051 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)
    (t0 : Float := 0.0)
    (t1 : Float := 0.1) :
    Except String SimulationResult :=
  solve? p x0 t0 t1 defaultInput

def simulateLqr? (p : QuadrotorParams := params)
    (cfg : LqrConfig := lqrConfig)
    (x0 : QuadrotorState :=
      mkState 0.15 (-0.12) 1.18 0.04 (-0.03) 0.05 0.0 0.0 0.0 0.0 0.0 0.0) :
    Except String LqrSimulationResult := do
  if !p.isValid then
    .error "quadrotor params are invalid"
  cfg.validate?
  if !stateIsValid x0 then
    .error "quadrotor LQR initial state is invalid"
  let samples := rolloutClosedLoop p (lqrController p cfg) cfg.stepSize cfg.steps x0
  if samples.isEmpty then
    .error "quadrotor LQR rollout produced no samples"
  let final := samples[samples.size - 1]!
  let finalInput := lqrController p cfg final
  let finalFullPhysics ← solveFullPhysics? p finalInput final
    "quadrotor final LQR full physics"
  let trace := DynamicEventTrace.empty.push (.interval (acceptedSegment 0.0 cfg.duration))
  trace.validate?
  pure {
    references := drakeReferences
    controllerName := "quadrotor-hover-lqr"
    t0 := 0.0
    t1 := cfg.duration
    stepSize := cfg.stepSize
    cost := lqrCostMetadata p cfg
    initialState := x0
    finalState := final
    samples := samples
    finalInput := finalInput
    initialEnergy := totalEnergy p x0
    finalEnergy := totalEnergy p final
    finalFullPhysics := finalFullPhysics
    trace := trace
    moves :=
      #[controllerMove 5300 "quadrotor LQR local linearization/controller"] ++
      trace.moves ++ #[finalFullPhysics.supportMove, finalFullPhysics.move]
  }

def buildEndToEnd? : Except String SimulationResult :=
  solvePassive?

end Tyr.EventSkeleton.Examples.Quadrotor
