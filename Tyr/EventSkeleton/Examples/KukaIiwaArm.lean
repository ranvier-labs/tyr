import Tyr.EventSkeleton.SceneGraph
import Tyr.EventSkeleton.Manipulator

/-!
# Drake Kuka Iiwa Arm Torque Controller Example

This ports the controller-side physics boundary of
`../drake/examples/kuka_iiwa_arm`.  Drake's `KukaTorqueController` sums four
joint-torque contributions:

* commanded feedforward torque,
* gravity compensation from the MultibodyPlant,
* virtual joint springs, and
* a state-dependent damper whose gains use the diagonal of the mass matrix.

The reusable primitive is `JointTorqueControllerInput.evaluate?`: examples can
provide mass-matrix diagonals and gravity compensation from a full multibody
compiler, a URDF-backed provider, or a fixture.  The algebraic controller is
kept exact and local in the event-skeleton graph.
-/

namespace Tyr.EventSkeleton.Examples.KukaIiwaArm

open Tyr.EventSkeleton

structure DrakeReference where
  path : String
  concept : String
  deriving Repr, BEq, Inhabited

def drakeReferences : Array DrakeReference :=
  #[
    {
      path := "../drake/examples/kuka_iiwa_arm/kuka_torque_controller.cc"
      concept := "builds the torque controller from gravity compensation, virtual springs, state-dependent damping, and commanded feedforward torque"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/kuka_torque_controller.h"
      concept := "declares estimated_state, desired_state, commanded_torque, and control ports"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/lcm_plan_interpolator.cc"
      concept := "adapts RobotPlanInterpolator to Iiwa LCM status, robot-plan, and command ports"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/lcm_plan_interpolator.h"
      concept := "declares LcmPlanInterpolator ports and required Initialize(plan_start_time, q0) call"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/iiwa_controller.cc"
      concept := "wires LcmPlanInterpolator to LCM subscribers/publisher and advances on status-message times"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/kuka_simulation.cc"
      concept := "builds an LCM-facing simulated hardware replacement with MultibodyPlant, SceneGraph, command receiver, controller, status sender, and Simulator.AdvanceTo"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/kuka_plan_runner.cc"
      concept := "waits for IIWA status, replaces the active plan on COMMITTED_ROBOT_PLAN, clears it on STOP, and publishes position commands sampled from a cubic trajectory"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/iiwa_lcm.h"
      concept := "forwards Iiwa LCM status and command sender/receiver systems"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/iiwa_lcm.cc"
      concept := "defines Drake's Iiwa LCM status and command sender/receiver wiring"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/test/kuka_torque_controller_test.cc"
      concept := "checks gravity-only, spring, and damping torque composition"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/iiwa_common.cc"
      concept := "defines Drake's position-control gains and torque-control stiffness/damping-ratio vectors"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/iiwa_common.h"
      concept := "declares Drake's common Iiwa gain helper functions"
    },
    {
      path := "../drake/manipulation/kuka_iiwa/iiwa_constants.cc"
      concept := "defines max joint velocity limits and the 5ms Iiwa LCM status period"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/move_iiwa_ee.cc"
      concept := "waits for measured status, solves an end-effector IK plan, and publishes COMMITTED_ROBOT_PLAN"
    },
    {
      path := "package://drake_models/iiwa_description/urdf/iiwa14_polytope_collision.urdf"
      concept := "the 7-DOF Iiwa model loaded by Drake's controller tests"
    },
    {
      path := "package://drake_models/iiwa_description/urdf/iiwa14_no_collision.urdf"
      concept := "the no-collision Iiwa model loaded by kuka_plan_runner for joint-name indexing"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/models/objects/block_for_pick_and_place_large_size.urdf"
      concept := "large pick-and-place cuboid with inset box collision and point-sphere contact probes"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/models/objects/block_for_pick_and_place_mid_size.urdf"
      concept := "mid-size pick-and-place cuboid with inset box collision and point-sphere contact probes"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/models/objects/folding_table.urdf"
      concept := "small folding-table support object with box tabletop collision"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/models/objects/round_table.urdf"
      concept := "round tabletop support object with cylinder collision"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/models/objects/simple_cuboid.urdf"
      concept := "small cuboid object with inset box collision and corner point-sphere probes"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/models/objects/simple_cylinder.urdf"
      concept := "cylindrical pick object with cylinder collision"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/models/objects/yellow_post.urdf"
      concept := "fixed post obstacle with base and post cylinder collision geometry"
    },
    {
      path := "../drake/examples/kuka_iiwa_arm/models/table/extra_heavy_duty_table_surface_only_collision.sdf"
      concept := "table model whose proximity geometry intentionally keeps only the tabletop surface"
    }
  ]

def numJoints : Nat := 7

def jointNames : Array String :=
  #["iiwa_joint_1", "iiwa_joint_2", "iiwa_joint_3", "iiwa_joint_4",
    "iiwa_joint_5", "iiwa_joint_6", "iiwa_joint_7"]

def positionCoordinateNames : Array String :=
  #["q0", "q1", "q2", "q3", "q4", "q5", "q6"]

def velocityCoordinateNames : Array String :=
  #["v0", "v1", "v2", "v3", "v4", "v5", "v6"]

def stateCoordinateNames : Array String :=
  positionCoordinateNames ++ velocityCoordinateNames

def torqueCoordinateNames : Array String :=
  #["tau0", "tau1", "tau2", "tau3", "tau4", "tau5", "tau6"]

def iiwaModelUrl : String :=
  "package://drake_models/iiwa_description/urdf/iiwa14_polytope_collision.urdf"

def iiwaNoCollisionModelUrl : String :=
  "package://drake_models/iiwa_description/urdf/iiwa14_no_collision.urdf"

def iiwaExampleModelRoot : String :=
  "../drake/examples/kuka_iiwa_arm/models"

inductive IiwaExampleModelKind where
  | metadata
  | object
  | table
  | desk
  deriving Repr, BEq, Inhabited

inductive IiwaExampleModelFormat where
  | urdf
  | sdf
  | bazel
  | markdown
  deriving Repr, BEq, Inhabited

namespace IiwaExampleModelFormat

def isPhysical : IiwaExampleModelFormat → Bool
  | .urdf => true
  | .sdf => true
  | .bazel => false
  | .markdown => false

def matchesPath (format : IiwaExampleModelFormat) (path : String) : Bool :=
  match format with
  | .urdf => path.endsWith ".urdf"
  | .sdf => path.endsWith ".sdf"
  | .bazel => path == "BUILD.bazel"
  | .markdown => path.endsWith ".md"

end IiwaExampleModelFormat

/-- Visual/proximity role of a primitive geometry entry copied from URDF/SDF. -/
inductive IiwaGeometryPrimitiveRole where
  | visual
  | collision
  deriving Repr, BEq, Inhabited

namespace IiwaGeometryPrimitiveRole

def sceneRole : IiwaGeometryPrimitiveRole → SceneGeometryRole
  | .visual => .illustration
  | .collision => .proximity

def isCollision : IiwaGeometryPrimitiveRole → Bool
  | .collision => true
  | .visual => false

end IiwaGeometryPrimitiveRole

structure IiwaGeometryPrimitive where
  name : String
  linkName : String
  role : IiwaGeometryPrimitiveRole
  X_LG : ScenePose3 := {}
  shape : SceneGeometryShape
  deriving Repr, BEq, Inhabited

namespace IiwaGeometryPrimitive

def validate? (geometry : IiwaGeometryPrimitive) : Except String Unit := do
  if geometry.name.isEmpty then
    .error "Iiwa geometry primitive name cannot be empty"
  if geometry.linkName.isEmpty then
    .error s!"Iiwa geometry primitive {geometry.name}: link name cannot be empty"
  geometry.X_LG.validate? s!"Iiwa geometry primitive {geometry.name} pose"
  geometry.shape.validate? s!"Iiwa geometry primitive {geometry.name} shape"

def isVisual (geometry : IiwaGeometryPrimitive) : Bool :=
  geometry.role == .visual

def isCollision (geometry : IiwaGeometryPrimitive) : Bool :=
  geometry.role == .collision

end IiwaGeometryPrimitive

private def vec3 (x y z : Float) : SceneVec3 :=
  { x := x, y := y, z := z }

private def translated (x y z : Float) : ScenePose3 :=
  ScenePose3.translated (vec3 x y z)

private def boxGeometry
    (role : IiwaGeometryPrimitiveRole) (name linkName : String)
    (sx sy sz : Float) (x : Float := 0.0) (y : Float := 0.0)
    (z : Float := 0.0) : IiwaGeometryPrimitive :=
  {
    name := name
    linkName := linkName
    role := role
    X_LG := translated x y z
    shape := .box sx sy sz
  }

private def cylinderGeometry
    (role : IiwaGeometryPrimitiveRole) (name linkName : String)
    (radius length : Float) (x : Float := 0.0) (y : Float := 0.0)
    (z : Float := 0.0) : IiwaGeometryPrimitive :=
  {
    name := name
    linkName := linkName
    role := role
    X_LG := translated x y z
    shape := .cylinder radius length
  }

private def sphereProbeGeometry
    (name linkName : String) (radius x y z : Float) : IiwaGeometryPrimitive :=
  {
    name := name
    linkName := linkName
    role := .collision
    X_LG := translated x y z
    shape := .sphere radius
  }

private def cuboidCornerProbeGeometry
    (linkName namePrefix : String) (hx hy hz : Float) :
    Array IiwaGeometryPrimitive := Id.run do
  let signs := #[-1.0, 1.0]
  let mut out : Array IiwaGeometryPrimitive := #[]
  let mut i := 0
  for sx in signs do
    for sy in signs do
      for sz in signs do
        out := out.push
          (sphereProbeGeometry s!"{namePrefix}_{i}" linkName 1.0e-7
            (sx * hx) (sy * hy) (sz * hz))
        i := i + 1
  return out

private def pickAndPlaceBlockGeometry
    (visualY collisionY probeY : Float) : Array IiwaGeometryPrimitive :=
  #[
    boxGeometry .visual "visual_box" "base_link" 0.06 visualY 0.2,
    boxGeometry .collision "inset_collision_box" "base_link" 0.059 collisionY 0.199
  ] ++ cuboidCornerProbeGeometry "base_link" "corner_probe" 0.03 probeY 0.1

private def blackBoxGeometry : Array IiwaGeometryPrimitive :=
  #[
    boxGeometry .visual "visual_box" "base_link" 0.165 0.055 0.180,
    boxGeometry .collision "inset_collision_box" "base_link" 0.164 0.054 0.170
  ] ++ cuboidCornerProbeGeometry "base_link" "corner_probe" 0.0825 0.0275 0.09 ++ #[
    sphereProbeGeometry "center_probe_0" "base_link" 1.0e-7 0.0 (-0.0275) (-0.09),
    sphereProbeGeometry "center_probe_1" "base_link" 1.0e-7 0.0 0.0275 (-0.09),
    sphereProbeGeometry "center_probe_2" "base_link" 1.0e-7 0.0 (-0.0275) 0.09,
    sphereProbeGeometry "center_probe_3" "base_link" 1.0e-7 0.0 0.0275 0.09
  ]

private def simpleCuboidGeometry : Array IiwaGeometryPrimitive :=
  #[
    boxGeometry .visual "visual_box" "base_link" 0.06 0.06 0.06,
    boxGeometry .collision "inset_collision_box" "base_link" 0.059 0.059 0.056
  ] ++ cuboidCornerProbeGeometry "base_link" "corner_probe" 0.03 0.03 0.03

private def openTopBoxGeometry : Array IiwaGeometryPrimitive :=
  #[
    boxGeometry .visual "front_visual" "bin" 0.01 0.52 0.3 0.19 0.0 0.15,
    boxGeometry .collision "front_collision" "bin" 0.01 0.52 0.3 0.19 0.0 0.15,
    boxGeometry .visual "back_visual" "bin" 0.01 0.52 0.3 (-0.19) 0.0 0.15,
    boxGeometry .collision "back_collision" "bin" 0.01 0.52 0.3 (-0.19) 0.0 0.15,
    boxGeometry .visual "left_visual" "bin" 0.38 0.01 0.3 0.0 0.26 0.15,
    boxGeometry .collision "left_collision" "bin" 0.38 0.01 0.3 0.0 0.26 0.15,
    boxGeometry .visual "right_visual" "bin" 0.38 0.01 0.3 0.0 (-0.26) 0.15,
    boxGeometry .collision "right_collision" "bin" 0.38 0.01 0.3 0.0 (-0.26) 0.15,
    boxGeometry .visual "bottom_visual" "bin" 0.38 0.52 0.01 0.0 0.0 0.0,
    boxGeometry .collision "bottom_collision" "bin" 0.38 0.52 0.01 0.0 0.0 (-0.005)
  ]

private def foldingTableGeometry : Array IiwaGeometryPrimitive :=
  #[
    boxGeometry .visual "tabletop_visual" "table_top" 0.48 0.37 0.015,
    boxGeometry .collision "tabletop_collision" "table_top" 0.48 0.37 0.015
  ]

private def roundTableGeometry : Array IiwaGeometryPrimitive :=
  #[
    cylinderGeometry .visual "tabletop_visual" "table_top" 0.30 0.017,
    cylinderGeometry .collision "tabletop_collision" "table_top" 0.30 0.017
  ]

private def simpleCylinderGeometry : Array IiwaGeometryPrimitive :=
  #[
    cylinderGeometry .visual "visual_cylinder" "cylinder_base" 0.0325 0.130,
    cylinderGeometry .collision "collision_cylinder" "cylinder_base" 0.0325 0.130
  ]

private def yellowPostGeometry : Array IiwaGeometryPrimitive :=
  #[
    cylinderGeometry .visual "base_visual" "cylinder_base" 0.177 0.065,
    cylinderGeometry .collision "base_collision" "cylinder_base" 0.177 0.065,
    cylinderGeometry .visual "post_visual" "post" 0.065 0.955,
    cylinderGeometry .collision "post_collision" "post" 0.065 0.955
  ]

private def tableBoxParts (role : IiwaGeometryPrimitiveRole) :
    Array IiwaGeometryPrimitive :=
  #[
    boxGeometry role "back_right_leg" "link" 0.05 0.05 0.762 (-0.33) (-0.35) 0.381,
    boxGeometry role "front_left_leg" "link" 0.05 0.05 0.762 0.33 0.35 0.381,
    boxGeometry role "left_crossbar" "link" 0.05 0.662 0.05 0.33 0.0 0.13335,
    boxGeometry role "right_crossbar" "link" 0.05 0.662 0.05 (-0.33) 0.0 0.13335,
    boxGeometry role "back_left_leg" "link" 0.05 0.05 0.762 (-0.33) 0.35 0.381,
    boxGeometry role "front_right_leg" "link" 0.05 0.05 0.762 0.33 (-0.35) 0.381,
    boxGeometry role "back_crossbar" "link" 0.6112 0.05 0.05 0.0 0.35 0.13335,
    boxGeometry role "front_crossbar" "link" 0.6112 0.05 0.05 0.0 (-0.35) 0.13335,
    boxGeometry role "surface" "link" 0.7112 0.762 0.057 0.0 0.0 0.736
  ]

private def extraHeavyDutyTableGeometry : Array IiwaGeometryPrimitive :=
  tableBoxParts .visual ++ tableBoxParts .collision

private def extraHeavyDutyTableSurfaceOnlyGeometry : Array IiwaGeometryPrimitive :=
  tableBoxParts .visual ++ #[
    boxGeometry .collision "surface" "link" 0.7112 0.762 0.057 0.0 0.0 0.736
  ]

/--
Asset facts copied from Drake's `examples/kuka_iiwa_arm/models` tree.

This is a provider boundary, not a physics backend: URDF/SDF parsing, contact
candidate generation, and plant dynamics still lower into the existing
`ParsedMultibodyPlantQuantities`, `ContactCandidateSet`, and
`FullPhysicsPrimitives` APIs.
-/
structure IiwaExampleModelAsset where
  relativePath : String
  format : IiwaExampleModelFormat
  kind : IiwaExampleModelKind
  modelName : String
  linkNames : Array String := #[]
  jointNames : Array String := #[]
  hasVisualGeometry : Bool := false
  hasCollisionGeometry : Bool := false
  hasInertial : Bool := false
  mass? : Option Float := none
  representativeShape : String := ""
  contactRole : String := ""
  deriving Repr, Inhabited

namespace IiwaExampleModelAsset

def fullPath (asset : IiwaExampleModelAsset) : String :=
  iiwaExampleModelRoot ++ "/" ++ asset.relativePath

def isPhysical (asset : IiwaExampleModelAsset) : Bool :=
  asset.format.isPhysical

def validate? (asset : IiwaExampleModelAsset) : Except String Unit := do
  if asset.relativePath.isEmpty then
    .error "Kuka model catalog asset path cannot be empty"
  if !asset.format.matchesPath asset.relativePath then
    .error s!"Kuka model asset {asset.relativePath}: format does not match path"
  if asset.modelName.isEmpty then
    .error s!"Kuka model asset {asset.relativePath}: modelName cannot be empty"
  match asset.mass? with
  | some mass =>
      if !mass.isFinite || mass <= 0.0 then
        .error s!"Kuka model asset {asset.relativePath}: mass must be positive and finite, got {mass}"
  | none => pure ()
  if asset.isPhysical then
    if asset.linkNames.isEmpty then
      .error s!"Kuka physical model asset {asset.relativePath}: link names cannot be empty"
    if !asset.hasInertial then
      .error s!"Kuka physical model asset {asset.relativePath}: inertial data must be recorded"
    if asset.mass?.isNone then
      .error s!"Kuka physical model asset {asset.relativePath}: mass must be recorded"
    if !asset.hasVisualGeometry && !asset.hasCollisionGeometry then
      .error s!"Kuka physical model asset {asset.relativePath}: visual or collision geometry must be recorded"
    if asset.hasCollisionGeometry && asset.contactRole.isEmpty then
      .error s!"Kuka physical model asset {asset.relativePath}: collision geometry needs a contact role"

end IiwaExampleModelAsset

def iiwaExampleModelAssets : Array IiwaExampleModelAsset :=
  #[
    {
      relativePath := "BUILD.bazel"
      format := .bazel
      kind := .metadata
      modelName := "models_filegroup"
      representativeShape := "public Bazel models_filegroup and install_data target"
    },
    {
      relativePath := "README.md"
      format := .markdown
      kind := .metadata
      modelName := "kuka_iiwa_arm_models_readme"
      representativeShape := "model catalog description for Kuka iiwa simulations"
    },
    {
      relativePath := "desk/transcendesk55inch.sdf"
      format := .sdf
      kind := .desk
      modelName := "transcendesk"
      linkNames := #["table_base", "table_upper"]
      jointNames := #["table_height"]
      hasVisualGeometry := true
      hasCollisionGeometry := true
      hasInertial := true
      mass? := some 83.5
      representativeShape := "height-adjustable desk boxes; base mass 30kg and upper mass 53.5kg"
      contactRole := "desk frame and tabletop collision boxes for manipulation scenes"
    },
    {
      relativePath := "objects/black_box.urdf"
      format := .urdf
      kind := .object
      modelName := "simple_cuboid"
      linkNames := #["base_link"]
      hasVisualGeometry := true
      hasCollisionGeometry := true
      hasInertial := true
      mass? := some 0.122
      representativeShape := "visual box 0.165 0.055 0.180; collision box 0.164 0.054 0.170 plus point-sphere probes"
      contactRole := "black cuboid pick object with box collision and point probes"
    },
    {
      relativePath := "objects/block_for_pick_and_place.urdf"
      format := .urdf
      kind := .object
      modelName := "simple_cuboid"
      linkNames := #["base_link"]
      hasVisualGeometry := true
      hasCollisionGeometry := true
      hasInertial := true
      mass? := some 0.1
      representativeShape := "visual box 0.06 0.06 0.2; collision box 0.059 0.059 0.199 plus eight point spheres"
      contactRole := "nominal pick-and-place block collision geometry"
    },
    {
      relativePath := "objects/block_for_pick_and_place_large_size.urdf"
      format := .urdf
      kind := .object
      modelName := "simple_cuboid"
      linkNames := #["base_link"]
      hasVisualGeometry := true
      hasCollisionGeometry := true
      hasInertial := true
      mass? := some 0.1
      representativeShape := "visual box 0.06 0.09 0.2; collision box 0.059 0.089 0.199 plus eight point spheres"
      contactRole := "large pick-and-place block collision geometry"
    },
    {
      relativePath := "objects/block_for_pick_and_place_mid_size.urdf"
      format := .urdf
      kind := .object
      modelName := "simple_cuboid"
      linkNames := #["base_link"]
      hasVisualGeometry := true
      hasCollisionGeometry := true
      hasInertial := true
      mass? := some 0.1
      representativeShape := "visual box 0.06 0.075 0.2; collision box 0.059 0.089 0.199 plus point spheres"
      contactRole := "mid-size pick-and-place block collision geometry"
    },
    {
      relativePath := "objects/folding_table.urdf"
      format := .urdf
      kind := .object
      modelName := "folding_table"
      linkNames := #["table_surface_center", "table_top"]
      jointNames := #["table_top_joint"]
      hasVisualGeometry := true
      hasCollisionGeometry := true
      hasInertial := true
      mass? := some 1.0
      representativeShape := "thin tabletop box 0.48 0.37 0.015"
      contactRole := "small tabletop support surface"
    },
    {
      relativePath := "objects/open_top_box.urdf"
      format := .urdf
      kind := .object
      modelName := "open_top_box"
      linkNames := #["bin"]
      hasVisualGeometry := true
      hasCollisionGeometry := true
      hasInertial := true
      mass? := some 0.122
      representativeShape := "five box panels: four walls and a floor"
      contactRole := "open bin collision geometry for contained-object contacts"
    },
    {
      relativePath := "objects/round_table.urdf"
      format := .urdf
      kind := .object
      modelName := "round_table"
      linkNames := #["table_surface_center", "table_top"]
      jointNames := #["table_top_joint"]
      hasVisualGeometry := true
      hasCollisionGeometry := true
      hasInertial := true
      mass? := some 1.0
      representativeShape := "cylinder tabletop length 0.017 radius 0.30"
      contactRole := "round tabletop support surface"
    },
    {
      relativePath := "objects/simple_cuboid.urdf"
      format := .urdf
      kind := .object
      modelName := "simple_cuboid"
      linkNames := #["base_link"]
      hasVisualGeometry := true
      hasCollisionGeometry := true
      hasInertial := true
      mass? := some 0.068
      representativeShape := "visual box 0.06 0.06 0.06; collision box 0.059 0.059 0.056 plus eight point spheres"
      contactRole := "small cuboid object with box collision and point probes"
    },
    {
      relativePath := "objects/simple_cylinder.urdf"
      format := .urdf
      kind := .object
      modelName := "simple_cylinder"
      linkNames := #["cylinder_base"]
      hasVisualGeometry := true
      hasCollisionGeometry := true
      hasInertial := true
      mass? := some 0.088
      representativeShape := "cylinder length 0.130 radius 0.0325"
      contactRole := "cylindrical pick object collision geometry"
    },
    {
      relativePath := "objects/yellow_post.urdf"
      format := .urdf
      kind := .object
      modelName := "yellow_post"
      linkNames := #["base", "cylinder_base", "post"]
      jointNames := #["base_joint", "post_joint"]
      hasVisualGeometry := true
      hasCollisionGeometry := true
      hasInertial := true
      mass? := some 20.0
      representativeShape := "base cylinder length 0.065 radius 0.177; post cylinder length 0.955 radius 0.065"
      contactRole := "fixed post obstacle with cylindrical collision geometry"
    },
    {
      relativePath := "table/extra_heavy_duty_table.sdf"
      format := .sdf
      kind := .table
      modelName := "extra_heavy_duty_table"
      linkNames := #["link"]
      hasVisualGeometry := true
      hasCollisionGeometry := true
      hasInertial := true
      mass? := some 53.5
      representativeShape := "table frame boxes plus tabletop collision surface"
      contactRole := "full table collision geometry for support and obstacle contacts"
    },
    {
      relativePath := "table/extra_heavy_duty_table_surface_only_collision.sdf"
      format := .sdf
      kind := .table
      modelName := "extra_heavy_duty_table_surface_only_collision"
      linkNames := #["link"]
      hasVisualGeometry := true
      hasCollisionGeometry := true
      hasInertial := true
      mass? := some 53.5
      representativeShape := "table visual frame with tabletop-only collision"
      contactRole := "tabletop-only collision support surface"
    }
  ]

private def hasDuplicateModelAssetPath (assets : Array IiwaExampleModelAsset) :
    Bool := Id.run do
  for i in [:assets.size] do
    for j in [:(assets.size - i - 1)] do
      let k := i + j + 1
      if assets[i]!.relativePath == assets[k]!.relativePath then
        return true
  return false

def iiwaExampleModelCatalogPaths : Array String :=
  iiwaExampleModelAssets.map (fun asset => asset.fullPath)

def iiwaPhysicalModelAssets : Array IiwaExampleModelAsset :=
  iiwaExampleModelAssets.filter (fun asset => asset.isPhysical)

def iiwaExampleObjectAssets : Array IiwaExampleModelAsset :=
  iiwaExampleModelAssets.filter (fun asset => asset.kind == .object)

def iiwaExampleSupportAssets : Array IiwaExampleModelAsset :=
  iiwaExampleModelAssets.filter (fun asset => asset.kind == .table || asset.kind == .desk)

def findIiwaExampleModelAsset? (path : String) :
    Option IiwaExampleModelAsset :=
  iiwaExampleModelAssets.find? (fun asset =>
    asset.relativePath == path || asset.fullPath == path)

private def requiredIiwaExampleModelPaths : Array String :=
  #[
    "BUILD.bazel",
    "README.md",
    "desk/transcendesk55inch.sdf",
    "objects/black_box.urdf",
    "objects/block_for_pick_and_place.urdf",
    "objects/block_for_pick_and_place_large_size.urdf",
    "objects/block_for_pick_and_place_mid_size.urdf",
    "objects/folding_table.urdf",
    "objects/open_top_box.urdf",
    "objects/round_table.urdf",
    "objects/simple_cuboid.urdf",
    "objects/simple_cylinder.urdf",
    "objects/yellow_post.urdf",
    "table/extra_heavy_duty_table.sdf",
    "table/extra_heavy_duty_table_surface_only_collision.sdf"
  ]

def validateIiwaExampleModelCatalog? : Except String Unit := do
  if iiwaExampleModelAssets.size != requiredIiwaExampleModelPaths.size then
    .error s!"Kuka model catalog size {iiwaExampleModelAssets.size} != expected {requiredIiwaExampleModelPaths.size}"
  if hasDuplicateModelAssetPath iiwaExampleModelAssets then
    .error "Kuka model catalog contains duplicate asset paths"
  for asset in iiwaExampleModelAssets do
    asset.validate?
  for path in requiredIiwaExampleModelPaths do
    match findIiwaExampleModelAsset? path with
    | some _ => pure ()
    | none => .error s!"Kuka model catalog missing required path {path}"
  if iiwaPhysicalModelAssets.size != 13 then
    .error s!"Kuka physical model catalog size {iiwaPhysicalModelAssets.size} != 13"
  if iiwaExampleObjectAssets.size != 10 then
    .error s!"Kuka object/support-object catalog size {iiwaExampleObjectAssets.size} != 10"
  if iiwaExampleSupportAssets.size != 3 then
    .error s!"Kuka support surface catalog size {iiwaExampleSupportAssets.size} != 3"

def iiwaModelPrimitiveGeometry (asset : IiwaExampleModelAsset) :
    Array IiwaGeometryPrimitive :=
  match asset.relativePath with
  | "objects/black_box.urdf" => blackBoxGeometry
  | "objects/block_for_pick_and_place.urdf" =>
      pickAndPlaceBlockGeometry 0.06 0.059 0.03
  | "objects/block_for_pick_and_place_large_size.urdf" =>
      pickAndPlaceBlockGeometry 0.09 0.089 0.045
  | "objects/block_for_pick_and_place_mid_size.urdf" =>
      pickAndPlaceBlockGeometry 0.075 0.089 0.0375
  | "objects/folding_table.urdf" => foldingTableGeometry
  | "objects/open_top_box.urdf" => openTopBoxGeometry
  | "objects/round_table.urdf" => roundTableGeometry
  | "objects/simple_cuboid.urdf" => simpleCuboidGeometry
  | "objects/simple_cylinder.urdf" => simpleCylinderGeometry
  | "objects/yellow_post.urdf" => yellowPostGeometry
  | "table/extra_heavy_duty_table.sdf" => extraHeavyDutyTableGeometry
  | "table/extra_heavy_duty_table_surface_only_collision.sdf" =>
      extraHeavyDutyTableSurfaceOnlyGeometry
  | "desk/transcendesk55inch.sdf" => extraHeavyDutyTableGeometry
  | _ => #[]

def iiwaModelCollisionGeometry (asset : IiwaExampleModelAsset) :
    Array IiwaGeometryPrimitive :=
  (iiwaModelPrimitiveGeometry asset).filter (fun geometry => geometry.isCollision)

def iiwaModelVisualGeometry (asset : IiwaExampleModelAsset) :
    Array IiwaGeometryPrimitive :=
  (iiwaModelPrimitiveGeometry asset).filter (fun geometry => geometry.isVisual)

def iiwaPrimitiveGeometryCatalog : Array IiwaGeometryPrimitive := Id.run do
  let mut out : Array IiwaGeometryPrimitive := #[]
  for asset in iiwaPhysicalModelAssets do
    out := out ++ iiwaModelPrimitiveGeometry asset
  return out

def validateIiwaExampleModelPrimitiveGeometry? : Except String Unit := do
  validateIiwaExampleModelCatalog?
  for asset in iiwaPhysicalModelAssets do
    let geometry := iiwaModelPrimitiveGeometry asset
    if geometry.isEmpty then
      .error s!"Kuka physical model asset {asset.relativePath}: primitive geometry is empty"
    for primitive in geometry do
      primitive.validate?
    if asset.hasVisualGeometry && (iiwaModelVisualGeometry asset).isEmpty then
      .error s!"Kuka physical model asset {asset.relativePath}: visual geometry flag has no visual primitive"
    if asset.hasCollisionGeometry && (iiwaModelCollisionGeometry asset).isEmpty then
      .error s!"Kuka physical model asset {asset.relativePath}: collision geometry flag has no collision primitive"

def iiwaLcmStatusPeriod : Float := 0.005

def iiwaMaxJointVelocities : Array Float :=
  #[1.483529, 1.483529, 1.745329, 1.308996, 2.268928, 2.356194, 2.356194]

structure IiwaState where
  q : Array Float
  v : Array Float
  deriving Repr, Inhabited

namespace IiwaState

def validate? (x : IiwaState) (label : String := "iiwa state") :
    Except String Unit := do
  if x.q.size != numJoints then
    .error s!"{label}: q size {x.q.size} != {numJoints}"
  if x.v.size != numJoints then
    .error s!"{label}: v size {x.v.size} != {numJoints}"
  for i in [:numJoints] do
    if !(x.q[i]!).isFinite then
      .error s!"{label}: q[{i}] must be finite, got {x.q[i]!}"
    if !(x.v[i]!).isFinite then
      .error s!"{label}: v[{i}] must be finite, got {x.v[i]!}"

def asArray (x : IiwaState) : Array Float :=
  x.q ++ x.v

end IiwaState

def drakeTestEstimatedState : IiwaState :=
  {
    q := #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    v := #[0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
  }

def drakeTestDesiredState : IiwaState :=
  {
    q := #[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7]
    v := #[-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1]
  }

def zeroTorque : Array Float :=
  Array.replicate numJoints 0.0

def torqueControlledGains : JointTorqueControllerGains :=
  {
    stiffness := #[1000.0, 1000.0, 1000.0, 500.0, 500.0, 500.0, 500.0]
    dampingRatio := Array.replicate numJoints 1.0
    label := "SetTorqueControlledIiwaGains"
  }

def positionControlledKp : Array Float :=
  Array.replicate numJoints 100.0

def positionControlledKi : Array Float :=
  Array.replicate numJoints 0.0

def positionControlledKd : Array Float :=
  positionControlledKp.map (fun kp => 2.0 * Float.sqrt kp)

def drakeTestSpringGains : JointTorqueControllerGains :=
  {
    stiffness := #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dampingRatio := Array.replicate numJoints 0.0
    label := "KukaTorqueControllerTest.SpringTorqueTest"
  }

def drakeTestDampingGains : JointTorqueControllerGains :=
  {
    stiffness := #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    dampingRatio := #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    label := "KukaTorqueControllerTest.DampingTorqueTest"
  }

structure IiwaMultibodyProviderData where
  modelUri : String
  baseFrameWelded : Bool := true
  massMatrixDiagonal : Array Float
  gravityCompensationTorque : Array Float
  label : String := ""
  deriving Repr, Inhabited

namespace IiwaMultibodyProviderData

def validate? (provider : IiwaMultibodyProviderData) : Except String Unit := do
  if provider.massMatrixDiagonal.size != numJoints then
    .error s!"iiwa provider {provider.label}: massMatrixDiagonal size {provider.massMatrixDiagonal.size} != {numJoints}"
  if provider.gravityCompensationTorque.size != numJoints then
    .error s!"iiwa provider {provider.label}: gravityCompensationTorque size {provider.gravityCompensationTorque.size} != {numJoints}"
  for i in [:numJoints] do
    let h := provider.massMatrixDiagonal[i]!
    let g := provider.gravityCompensationTorque[i]!
    if !h.isFinite || h < 0.0 then
      .error s!"iiwa provider {provider.label}: massMatrixDiagonal[{i}] must be nonnegative and finite, got {h}"
    if !g.isFinite then
      .error s!"iiwa provider {provider.label}: gravityCompensationTorque[{i}] must be finite, got {g}"

end IiwaMultibodyProviderData

/--
Representative output of the multibody side of Drake's controller test.

The controller primitive only requires the mass-matrix diagonal and gravity
compensation torque.  A future URDF/multibody compiler can replace this
provider while preserving the same controller interface.
-/
def drakeTestProvider : IiwaMultibodyProviderData :=
  {
    modelUri := iiwaModelUrl
    baseFrameWelded := true
    massMatrixDiagonal := #[4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]
    gravityCompensationTorque := #[12.0, -8.0, 5.0, -3.0, 1.5, -0.75, 0.25]
    label := "iiwa14-polytope-collision-test-provider"
  }

def controllerInput
    (provider : IiwaMultibodyProviderData := drakeTestProvider)
    (estimated : IiwaState := drakeTestEstimatedState)
    (desired : IiwaState := drakeTestDesiredState)
    (commandedTorque : Array Float := zeroTorque) :
    JointTorqueControllerInput :=
  {
    estimatedState := estimated.asArray
    desiredState := desired.asArray
    commandedTorque := commandedTorque
    gravityCompensationTorque := provider.gravityCompensationTorque
    massMatrixDiagonal := provider.massMatrixDiagonal
    label := "iiwa torque controller input"
  }

def evaluateTorqueController?
    (gains : JointTorqueControllerGains := torqueControlledGains)
    (provider : IiwaMultibodyProviderData := drakeTestProvider)
    (estimated : IiwaState := drakeTestEstimatedState)
    (desired : IiwaState := drakeTestDesiredState)
    (commandedTorque : Array Float := zeroTorque) :
    Except String JointTorqueControllerOutput := do
  provider.validate?
  estimated.validate? "estimated iiwa state"
  desired.validate? "desired iiwa state"
  (controllerInput provider estimated desired commandedTorque).evaluate? gains

structure IiwaTorqueControlPhysicsState where
  provider : IiwaMultibodyProviderData := drakeTestProvider
  estimated : IiwaState := drakeTestEstimatedState
  desired : IiwaState := drakeTestDesiredState
  commandedTorque : Array Float := zeroTorque
  deriving Repr, Inhabited

namespace IiwaTorqueControlPhysicsState

def validate? (snapshot : IiwaTorqueControlPhysicsState)
    (gains : JointTorqueControllerGains := torqueControlledGains) :
    Except String Unit := do
  snapshot.provider.validate?
  snapshot.estimated.validate? "estimated iiwa state"
  snapshot.desired.validate? "desired iiwa state"
  if snapshot.commandedTorque.size != numJoints then
    .error s!"iiwa torque-control snapshot: commandedTorque size {snapshot.commandedTorque.size} != {numJoints}"
  for i in [:numJoints] do
    let tau := snapshot.commandedTorque[i]!
    if !tau.isFinite then
      .error s!"iiwa torque-control snapshot: commandedTorque[{i}] must be finite, got {tau}"
  (controllerInput snapshot.provider snapshot.estimated snapshot.desired
    snapshot.commandedTorque).validate? gains

def controllerOutput? (snapshot : IiwaTorqueControlPhysicsState)
    (gains : JointTorqueControllerGains := torqueControlledGains) :
    Except String JointTorqueControllerOutput := do
  snapshot.validate? gains
  evaluateTorqueController? gains snapshot.provider snapshot.estimated
    snapshot.desired snapshot.commandedTorque

end IiwaTorqueControlPhysicsState

def torqueControlPhysicsState
    (provider : IiwaMultibodyProviderData := drakeTestProvider)
    (estimated : IiwaState := drakeTestEstimatedState)
    (desired : IiwaState := drakeTestDesiredState)
    (commandedTorque : Array Float := zeroTorque) :
    IiwaTorqueControlPhysicsState :=
  {
    provider := provider
    estimated := estimated
    desired := desired
    commandedTorque := commandedTorque
  }

def gravityOnlyGains : JointTorqueControllerGains :=
  {
    stiffness := Array.replicate numJoints 0.0
    dampingRatio := Array.replicate numJoints 0.0
    label := "gravity-only iiwa controller"
  }

def controllerGraph : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 8100, kind := .state .boundary, label := "estimated_state" }
    |>.addVertex { id := 8101, kind := .state .boundary, label := "desired_state" }
    |>.addVertex { id := 8102, kind := .state .boundary, label := "commanded_torque" }
    |>.addVertex { id := 8103, kind := .state .interior, label := "mass_matrix_diagonal" }
    |>.addVertex { id := 8104, kind := .state .interior, label := "gravity_compensation_torque" }
    |>.addVertex { id := 8105, kind := .state .boundary, label := "control_torque" }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8103, 8104]
      reads := #[8100, 8101, 8102, 8103, 8104]
      writes := #[8105]
      label := "kuka-iiwa-state-dependent-torque-controller"
    }

def fullPhysicsIntervalVertex : VertexId := 8106

def closedLoopPhysicsGraph : SkeletonGraph :=
  controllerGraph
    |>.addVertex { id := fullPhysicsIntervalVertex, kind := .interval, label := "iiwa full physics plant interval" }
    |>.addVertex { id := 8107, kind := .state .checkpoint, label := "next iiwa state" }
    |>.addMove {
      kind := .intervalAdjoint
      targets := #[fullPhysicsIntervalVertex]
      reads := #[8100, 8103, 8104, 8105]
      writes := #[8107]
      label := "full physics primitive: M(q) vdot = tau - bias"
    }

def iiwaFullPlantModel : FullMultibodyPlantModel :=
  {
    modelName := "iiwa14"
    modelUri := iiwaModelUrl
    numPositions := numJoints
    numVelocities := numJoints
    numActuatedDofs := numJoints
    label := "iiwa14-polytope-collision full plant"
  }

def iiwaFullPlantConfig : MultibodyPlantConfigPrimitive :=
  {
    timeStep := iiwaLcmStatusPeriod
    penetrationAllowance := 1.0e-3
    stictionTolerance := 1.0e-3
    contactApproximation := .sap
  }

def iiwaActuationMap : GeneralizedActuationMap :=
  GeneralizedActuationMap.identity numJoints "Iiwa fixed-base joint actuation map"

def iiwaGeneralizedActuation? (actuation : Array Float) :
    Except String (Array Float) :=
  iiwaActuationMap.generalizedForces? actuation

def fullPhysicsPrimitivesFromController?
    (provider : IiwaMultibodyProviderData := drakeTestProvider)
    (estimated : IiwaState := drakeTestEstimatedState)
    (controllerOutput : JointTorqueControllerOutput) :
    Except String FullPhysicsPrimitives := do
  provider.validate?
  estimated.validate? "estimated iiwa state"
  if controllerOutput.controlTorque.size != numJoints then
    .error s!"iiwa full physics: control torque size {controllerOutput.controlTorque.size} != {numJoints}"
  let generalizedActuation ← iiwaGeneralizedActuation? controllerOutput.controlTorque
  pure {
    massMatrix := FloatMatrix.diagonal provider.massMatrixDiagonal
    qdot := estimated.v
    actuationForces := generalizedActuation
    biasForces := provider.gravityCompensationTorque
    contactCandidates := #[]
    supportPolicy := .fullSupport
    contactForceSource := .precomputed
    contactForces := #[]
    label := "iiwa torque-control full physics primitive"
  }

def fullPhysicsPrimitiveProvider
    (gains : JointTorqueControllerGains := torqueControlledGains)
    (label : String := "iiwa torque-control full physics provider") :
    FullPhysicsPrimitiveProvider IiwaTorqueControlPhysicsState :=
  {
    label := label
    primitivesAt? := fun snapshot => do
      let controllerOutput ← snapshot.controllerOutput? gains
      let primitives ← fullPhysicsPrimitivesFromController? snapshot.provider
        snapshot.estimated controllerOutput
      pure { primitives with label := label }
  }

def fullPhysicsEquationFromController?
    (provider : IiwaMultibodyProviderData := drakeTestProvider)
    (estimated : IiwaState := drakeTestEstimatedState)
    (controllerOutput : JointTorqueControllerOutput) :
    Except String FullPhysicsEquation := do
  let primitives ← fullPhysicsPrimitivesFromController? provider estimated controllerOutput
  primitives.equation?

def fullPlantStepFromController?
    (provider : IiwaMultibodyProviderData := drakeTestProvider)
    (estimated : IiwaState := drakeTestEstimatedState)
    (controllerOutput : JointTorqueControllerOutput) :
    Except String FullMultibodyPlantStep := do
  provider.validate?
  estimated.validate? "estimated iiwa state"
  if controllerOutput.controlTorque.size != numJoints then
    .error s!"iiwa full plant step: control torque size {controllerOutput.controlTorque.size} != {numJoints}"
  pure {
    model := iiwaFullPlantModel
    config := iiwaFullPlantConfig
    q0 := estimated.q
    v0 := estimated.v
    actuation := controllerOutput.controlTorque
    t0 := 0.0
    t1 := iiwaLcmStatusPeriod
    label := "iiwa torque-control full plant step"
  }

def solveFullPhysics?
    (gains : JointTorqueControllerGains := torqueControlledGains)
    (provider : IiwaMultibodyProviderData := drakeTestProvider)
    (estimated : IiwaState := drakeTestEstimatedState)
    (desired : IiwaState := drakeTestDesiredState)
    (commandedTorque : Array Float := zeroTorque) :
    Except String (JointTorqueControllerOutput × FullPhysicsResult × FullMultibodyPlantStep) := do
  let snapshot := torqueControlPhysicsState provider estimated desired commandedTorque
  let controllerOutput ← snapshot.controllerOutput? gains
  let fullPhysics ←
    (fullPhysicsPrimitiveProvider gains).solveAt? snapshot fullPhysicsIntervalVertex
  let step ← fullPlantStepFromController? provider estimated controllerOutput
  step.validate?
  pure (controllerOutput, fullPhysics, step)

inductive IiwaPlanInterpolatorType where
  | zeroOrderHold
  | firstOrderHold
  | cubic
  | pchip
  deriving Repr, BEq, Inhabited

namespace IiwaPlanInterpolatorType

def flag : IiwaPlanInterpolatorType → String
  | .zeroOrderHold => "zoh"
  | .firstOrderHold => "foh"
  | .cubic => "cubic"
  | .pchip => "pchip"

def drakeName : IiwaPlanInterpolatorType → String
  | .zeroOrderHold => "InterpolatorType::ZeroOrderHold"
  | .firstOrderHold => "InterpolatorType::FirstOrderHold"
  | .cubic => "InterpolatorType::Cubic"
  | .pchip => "InterpolatorType::Pchip"

end IiwaPlanInterpolatorType

structure IiwaLcmChannels where
  statusChannel : String := "IIWA_STATUS"
  commandChannel : String := "IIWA_COMMAND"
  planChannel : String := "COMMITTED_ROBOT_PLAN"
  deriving Repr, Inhabited

namespace IiwaLcmChannels

def validate? (channels : IiwaLcmChannels) : Except String Unit := do
  if channels.statusChannel == "" then
    .error "Iiwa status channel must be nonempty"
  if channels.commandChannel == "" then
    .error "Iiwa command channel must be nonempty"
  if channels.planChannel == "" then
    .error "Iiwa robot-plan channel must be nonempty"
  if channels.statusChannel == channels.commandChannel ||
      channels.statusChannel == channels.planChannel ||
      channels.commandChannel == channels.planChannel then
    .error "Iiwa LCM status, command, and plan channels must be distinct"

end IiwaLcmChannels

def iiwaLcmChannels : IiwaLcmChannels := {}

structure LcmPlanInterpolatorBoundary where
  modelUri : String := iiwaModelUrl
  channels : IiwaLcmChannels := iiwaLcmChannels
  interpolatorType : IiwaPlanInterpolatorType := .cubic
  numPositions : Nat := numJoints
  numVelocities : Nat := numJoints
  stateInputPort : String := "status_receiver_lcmt_iiwa_status"
  planInputPort : String := "plan_interpolator_plan"
  commandOutputPort : String := "command_sender_lcmt_iiwa_command"
  defaultPlanUpdateInterval : Float := 0.1
  statusPeriod : Float := iiwaLcmStatusPeriod
  initializeFromFirstStatus : Bool := true
  holdPlanStartTimeSource : String := "first lcmt_iiwa_status utime"
  holdPlanQ0Source : String := "first lcmt_iiwa_status joint_position_measured"
  deriving Repr, Inhabited

namespace LcmPlanInterpolatorBoundary

def validate? (boundary : LcmPlanInterpolatorBoundary) : Except String Unit := do
  if boundary.modelUri == "" then
    .error "LcmPlanInterpolator model URI must be nonempty"
  boundary.channels.validate?
  if boundary.numPositions != numJoints || boundary.numVelocities != numJoints then
    .error s!"LcmPlanInterpolator dimensions should be {numJoints}/{numJoints}, got {boundary.numPositions}/{boundary.numVelocities}"
  if boundary.stateInputPort == "" || boundary.planInputPort == "" ||
      boundary.commandOutputPort == "" then
    .error "LcmPlanInterpolator exported port names must be nonempty"
  if !boundary.defaultPlanUpdateInterval.isFinite ||
      boundary.defaultPlanUpdateInterval <= 0.0 then
    .error s!"LcmPlanInterpolator default plan update interval must be positive, got {boundary.defaultPlanUpdateInterval}"
  if !boundary.statusPeriod.isFinite || boundary.statusPeriod <= 0.0 then
    .error s!"Iiwa status period must be positive and finite, got {boundary.statusPeriod}"
  if !boundary.initializeFromFirstStatus then
    .error "LcmPlanInterpolator runtime boundary must initialize from first status before output"

end LcmPlanInterpolatorBoundary

def lcmPlanInterpolatorBoundary : LcmPlanInterpolatorBoundary := {}

inductive KukaSimulationControlMode where
  | positionControl
  | torqueControl
  deriving Repr, BEq, Inhabited

namespace KukaSimulationControlMode

def label : KukaSimulationControlMode → String
  | .positionControl => "position-control"
  | .torqueControl => "torque-control"

def controllerSystem : KukaSimulationControlMode → String
  | .positionControl => "InverseDynamicsController"
  | .torqueControl => "KukaTorqueController"

def usesCommandedTorqueInput : KukaSimulationControlMode → Bool
  | .positionControl => false
  | .torqueControl => true

end KukaSimulationControlMode

structure KukaSimulationBoundary where
  modelUri : String := iiwaModelUrl
  channels : IiwaLcmChannels := iiwaLcmChannels
  simDt : Float := 3.0e-3
  targetRealtimeRate : Float := 1.0
  simulationSec? : Option Float := none
  controlMode : KukaSimulationControlMode := .positionControl
  numIiwa : Nat := 1
  statusPeriod : Float := iiwaLcmStatusPeriod
  weldsBaseToWorld : Bool := true
  usesSceneGraphVisualization : Bool := true
  addsFloor : Bool := false
  commandSubscriberName : String := "command_subscriber"
  commandReceiverName : String := "command_receiver"
  desiredStateInterpolatorName : String := "desired_state_from_position"
  statusPublisherName : String := "status_publisher"
  statusSenderName : String := "status_sender"
  deriving Repr, Inhabited

namespace KukaSimulationBoundary

def validate? (boundary : KukaSimulationBoundary) : Except String Unit := do
  if boundary.modelUri == "" then
    .error "kuka_simulation model URI must be nonempty"
  boundary.channels.validate?
  if !boundary.simDt.isFinite || boundary.simDt <= 0.0 then
    .error s!"kuka_simulation sim_dt must be positive and finite, got {boundary.simDt}"
  if !boundary.targetRealtimeRate.isFinite || boundary.targetRealtimeRate < 0.0 then
    .error s!"kuka_simulation target realtime rate must be finite and nonnegative, got {boundary.targetRealtimeRate}"
  match boundary.simulationSec? with
  | some seconds =>
      if !seconds.isFinite || seconds < 0.0 then
        .error s!"kuka_simulation simulation_sec must be finite and nonnegative when supplied, got {seconds}"
  | none => pure ()
  if boundary.numIiwa == 0 then
    .error "kuka_simulation must model at least one iiwa arm"
  if !boundary.statusPeriod.isFinite || boundary.statusPeriod <= 0.0 then
    .error s!"kuka_simulation status period must be positive and finite, got {boundary.statusPeriod}"
  if !boundary.weldsBaseToWorld then
    .error "kuka_simulation should weld the iiwa base to world"
  if !boundary.usesSceneGraphVisualization then
    .error "kuka_simulation should include SceneGraph visualization wiring"
  if boundary.addsFloor then
    .error "kuka_simulation source still has the floor as a TODO"
  if boundary.commandSubscriberName == "" || boundary.commandReceiverName == "" ||
      boundary.desiredStateInterpolatorName == "" || boundary.statusPublisherName == "" ||
      boundary.statusSenderName == "" then
    .error "kuka_simulation named systems must be nonempty"

def controllerSystem (boundary : KukaSimulationBoundary) : String :=
  boundary.controlMode.controllerSystem

def usesCommandedTorqueInput (boundary : KukaSimulationBoundary) : Bool :=
  boundary.controlMode.usesCommandedTorqueInput

end KukaSimulationBoundary

def kukaSimulationBoundary : KukaSimulationBoundary := {}

structure KukaPlanRunnerBoundary where
  modelUri : String := iiwaNoCollisionModelUrl
  channels : IiwaLcmChannels := iiwaLcmChannels
  stopChannel : String := "STOP"
  jointCount : Nat := numJoints
  waitsForFirstStatus : Bool := true
  minKnotPoints : Nat := 2
  firstKnotSource : String := "status.joint_position_commanded"
  interpolation : String := "PiecewisePolynomial::CubicWithContinuousSecondDerivatives"
  replacesActivePlan : Bool := true
  stopDiscardsPlan : Bool := true
  ignoresUnknownJointNames : Bool := true
  commandPositionField : String := "joint_position"
  commandTimestampSource : String := "status.utime"
  deriving Repr, Inhabited

namespace KukaPlanRunnerBoundary

def validate? (boundary : KukaPlanRunnerBoundary) : Except String Unit := do
  if boundary.modelUri == "" then
    .error "kuka_plan_runner model URI must be nonempty"
  boundary.channels.validate?
  if boundary.stopChannel == "" then
    .error "kuka_plan_runner stop channel must be nonempty"
  if boundary.stopChannel == boundary.channels.statusChannel ||
      boundary.stopChannel == boundary.channels.commandChannel ||
      boundary.stopChannel == boundary.channels.planChannel then
    .error "kuka_plan_runner STOP channel must be distinct from status, command, and plan channels"
  if boundary.jointCount != numJoints then
    .error s!"kuka_plan_runner should expose {numJoints} joints, got {boundary.jointCount}"
  if !boundary.waitsForFirstStatus then
    .error "kuka_plan_runner must wait for at least one IIWA status message"
  if boundary.minKnotPoints < 2 then
    .error "kuka_plan_runner must reject plans with fewer than two knot points"
  if boundary.firstKnotSource != "status.joint_position_commanded" then
    .error s!"kuka_plan_runner first knot should come from commanded status, got {boundary.firstKnotSource}"
  if boundary.interpolation == "" then
    .error "kuka_plan_runner interpolation primitive must be named"
  if !boundary.replacesActivePlan then
    .error "kuka_plan_runner should replace any active plan when a new plan arrives"
  if !boundary.stopDiscardsPlan then
    .error "kuka_plan_runner STOP handling should discard the active plan"
  if boundary.commandPositionField != "joint_position" then
    .error s!"kuka_plan_runner should write joint_position commands, got {boundary.commandPositionField}"
  if boundary.commandTimestampSource != "status.utime" then
    .error s!"kuka_plan_runner command timestamps should come from status.utime, got {boundary.commandTimestampSource}"

end KukaPlanRunnerBoundary

def kukaPlanRunnerBoundary : KukaPlanRunnerBoundary := {}

structure IiwaRigidPose where
  xyz : Array Float := #[0.0, 0.0, 0.0]
  rpy : Array Float := #[0.0, 0.0, 0.0]
  deriving Repr, Inhabited

namespace IiwaRigidPose

private def finiteVector (xs : Array Float) : Bool :=
  xs.all (fun x => x.isFinite)

def validate? (pose : IiwaRigidPose) : Except String Unit := do
  if pose.xyz.size != 3 then
    .error s!"Iiwa pose xyz size {pose.xyz.size} != 3"
  if pose.rpy.size != 3 then
    .error s!"Iiwa pose rpy size {pose.rpy.size} != 3"
  if !finiteVector pose.xyz || !finiteVector pose.rpy then
    .error "Iiwa pose entries must be finite"

end IiwaRigidPose

structure MoveIiwaEeBoundary where
  modelUri : String := iiwaModelUrl
  channels : IiwaLcmChannels :=
    { iiwaLcmChannels with commandChannel := "IIWA_COMMAND" }
  baseFrame : String := "base"
  endEffectorFrame : String := "iiwa_link_ee"
  targetPose : IiwaRigidPose := {}
  ikSamples : Nat := 100
  jointVelocityLimits : Array Float := iiwaMaxJointVelocities
  waitsForStatus : Bool := true
  statusPositionField : String := "joint_position_measured"
  planFailureIsNonfatal : Bool := true
  deriving Repr, Inhabited

namespace MoveIiwaEeBoundary

private def finitePositiveVector (xs : Array Float) : Bool :=
  xs.all (fun x => x.isFinite && x > 0.0)

def validate? (boundary : MoveIiwaEeBoundary) : Except String Unit := do
  if boundary.modelUri == "" then
    .error "move_iiwa_ee model URI must be nonempty"
  boundary.channels.validate?
  if boundary.baseFrame == "" || boundary.endEffectorFrame == "" then
    .error "move_iiwa_ee base and end-effector frame names must be nonempty"
  boundary.targetPose.validate?
  if boundary.ikSamples == 0 then
    .error "move_iiwa_ee IK sample count must be positive"
  if boundary.jointVelocityLimits.size != numJoints ||
      !finitePositiveVector boundary.jointVelocityLimits then
    .error s!"move_iiwa_ee joint velocity limits must have {numJoints} finite positive entries"
  if !boundary.waitsForStatus then
    .error "move_iiwa_ee must wait for a measured status message before planning"
  if boundary.statusPositionField != "joint_position_measured" then
    .error s!"move_iiwa_ee should read joint_position_measured, got {boundary.statusPositionField}"

end MoveIiwaEeBoundary

def moveIiwaEeBoundary : MoveIiwaEeBoundary := {}

def kukaRuntimeGraph
    (plan : LcmPlanInterpolatorBoundary := lcmPlanInterpolatorBoundary)
    (ee : MoveIiwaEeBoundary := moveIiwaEeBoundary)
    (simulation : KukaSimulationBoundary := kukaSimulationBoundary)
    (runner : KukaPlanRunnerBoundary := kukaPlanRunnerBoundary) : SkeletonGraph :=
  SkeletonGraph.empty
    |>.addVertex { id := 8120, kind := .state .boundary, label := plan.channels.statusChannel }
    |>.addVertex { id := 8121, kind := .state .boundary, label := plan.channels.planChannel }
    |>.addVertex { id := 8122, kind := .state .interior, label := "RobotPlanInterpolator state" }
    |>.addVertex { id := 8123, kind := .state .boundary, label := plan.channels.commandChannel }
    |>.addVertex { id := 8124, kind := .state .boundary, label := "move_iiwa_ee target pose" }
    |>.addVertex { id := 8125, kind := .state .boundary, label := "published lcmt_robot_plan" }
    |>.addVertex { id := 8126, kind := .interval, label := "kuka_simulation MultibodyPlant interval" }
    |>.addVertex { id := 8127, kind := .state .interior, label := "kuka_simulation controller and status sender" }
    |>.addVertex { id := 8128, kind := .state .interior, label := "kuka_plan_runner active plan" }
    |>.addVertex { id := 8129, kind := .state .boundary, label := runner.stopChannel }
    |>.addMove {
      kind := .freezeControl
      targets := #[8122]
      reads := #[8120]
      writes := #[8122]
      label := s!"initialize hold plan from {plan.holdPlanQ0Source}"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8122]
      reads := #[8120, 8121, 8122]
      writes := #[8123]
      label := s!"LcmPlanInterpolator {plan.interpolatorType.flag} plan-to-command adapter"
    }
    |>.addMove {
      kind := .clockedUpdate
      targets := #[8123]
      reads := #[8120, 8122]
      writes := #[8123]
      label := s!"advance on Iiwa status time and publish command every status message ({plan.statusPeriod}s nominal)"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8125]
      reads := #[8120, 8124]
      writes := #[8125]
      cost := { work := ee.ikSamples.toFloat }
      label := s!"move_iiwa_ee IK plan to {ee.endEffectorFrame} using {ee.ikSamples} samples"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8127]
      reads := #[8123, 8126]
      writes := #[8127]
      label := s!"kuka_simulation {simulation.controlMode.label} {simulation.controllerSystem} command-to-actuation block"
    }
    |>.addMove {
      kind := .intervalAdjoint
      targets := #[8126]
      reads := #[8126, 8127]
      writes := #[8126]
      label := s!"kuka_simulation MultibodyPlant AdvanceTo with sim_dt={simulation.simDt}"
    }
    |>.addMove {
      kind := .clockedUpdate
      targets := #[8120]
      reads := #[8126, 8127]
      writes := #[8120]
      label := s!"kuka_simulation publishes lcmt_iiwa_status every {simulation.statusPeriod}s"
    }
    |>.addMove {
      kind := .freezeControl
      targets := #[8128]
      reads := #[8120]
      writes := #[8128]
      label := "kuka_plan_runner waits for first IIWA status before activating plans"
    }
    |>.addMove {
      kind := .localSchurBlock
      targets := #[8128]
      reads := #[8120, 8121]
      writes := #[8128]
      label := s!"kuka_plan_runner replaces active plan with {runner.interpolation}"
    }
    |>.addMove {
      kind := .resetTranspose
      targets := #[8128]
      reads := #[8129]
      writes := #[8128]
      label := "kuka_plan_runner STOP reset discards active plan"
    }
    |>.addMove {
      kind := .clockedUpdate
      targets := #[8123]
      reads := #[8120, 8128]
      writes := #[8123]
      label := s!"kuka_plan_runner samples active plan at {runner.commandTimestampSource} and publishes {runner.commandPositionField}"
    }

structure KukaIiwaRuntimeResult where
  references : Array DrakeReference
  modelCatalog : Array IiwaExampleModelAsset
  channels : IiwaLcmChannels
  planInterpolator : LcmPlanInterpolatorBoundary
  moveEndEffector : MoveIiwaEeBoundary
  simulation : KukaSimulationBoundary
  planRunner : KukaPlanRunnerBoundary
  graph : SkeletonGraph
  deriving Repr, Inhabited

def buildRuntimeBoundaries?
    (plan : LcmPlanInterpolatorBoundary := lcmPlanInterpolatorBoundary)
    (ee : MoveIiwaEeBoundary := moveIiwaEeBoundary)
    (simulation : KukaSimulationBoundary := kukaSimulationBoundary)
    (runner : KukaPlanRunnerBoundary := kukaPlanRunnerBoundary) :
    Except String KukaIiwaRuntimeResult := do
  plan.validate?
  ee.validate?
  simulation.validate?
  runner.validate?
  validateIiwaExampleModelCatalog?
  validateIiwaExampleModelPrimitiveGeometry?
  if plan.modelUri != ee.modelUri then
    .error "Kuka plan interpolator and move_iiwa_ee should use the same Iiwa model"
  if plan.channels.statusChannel != ee.channels.statusChannel ||
      plan.channels.planChannel != ee.channels.planChannel then
    .error "Kuka plan interpolator and move_iiwa_ee must agree on status and plan channels"
  if simulation.channels.statusChannel != plan.channels.statusChannel ||
      simulation.channels.commandChannel != plan.channels.commandChannel then
    .error "kuka_simulation must agree with runtime status and command channels"
  if runner.channels.statusChannel != plan.channels.statusChannel ||
      runner.channels.commandChannel != plan.channels.commandChannel ||
      runner.channels.planChannel != plan.channels.planChannel then
    .error "kuka_plan_runner must agree with runtime status, command, and plan channels"
  let graph := kukaRuntimeGraph plan ee simulation runner
  pure {
    references := drakeReferences
    modelCatalog := iiwaExampleModelAssets
    channels := plan.channels
    planInterpolator := plan
    moveEndEffector := ee
    simulation := simulation
    planRunner := runner
    graph := graph
  }

structure KukaIiwaResult where
  references : Array DrakeReference
  modelCatalog : Array IiwaExampleModelAsset
  provider : IiwaMultibodyProviderData
  gains : JointTorqueControllerGains
  controllerOutput : JointTorqueControllerOutput
  fullPhysics : FullPhysicsResult
  plantStep : FullMultibodyPlantStep
  graph : SkeletonGraph
  deriving Repr, Inhabited

def buildEndToEnd? : Except String KukaIiwaResult := do
  validateIiwaExampleModelCatalog?
  validateIiwaExampleModelPrimitiveGeometry?
  let (output, fullPhysics, plantStep) ← solveFullPhysics? torqueControlledGains
  pure {
    references := drakeReferences
    modelCatalog := iiwaExampleModelAssets
    provider := drakeTestProvider
    gains := torqueControlledGains
    controllerOutput := output
    fullPhysics := fullPhysics
    plantStep := plantStep
    graph := closedLoopPhysicsGraph
  }

end Tyr.EventSkeleton.Examples.KukaIiwaArm
