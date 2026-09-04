import Tyr.EventSkeleton.Core
import Tyr.EventSkeleton.Interval
import Tyr.EventSkeleton.Saltation
import Tyr.EventSkeleton.Mark
import Tyr.EventSkeleton.Branch
import Tyr.EventSkeleton.Trace
import Tyr.EventSkeleton.Physics
import Tyr.EventSkeleton.Manipulator
import Tyr.EventSkeleton.Contact
import Tyr.EventSkeleton.NamedVector
import Tyr.EventSkeleton.SceneGraph
import Tyr.EventSkeleton.HardwareSim
import Tyr.EventSkeleton.Examples.DrakeExampleCatalog
import Tyr.EventSkeleton.Examples.DrakeWitness
import Tyr.EventSkeleton.Examples.UrdfContact
import Tyr.EventSkeleton.Examples.CylinderMulticontact
import Tyr.EventSkeleton.Examples.BouncingBall
import Tyr.EventSkeleton.Examples.Pendulum
import Tyr.EventSkeleton.Examples.Acrobot
import Tyr.EventSkeleton.Examples.CartPole
import Tyr.EventSkeleton.Examples.Quadrotor
import Tyr.EventSkeleton.Examples.RollingSphere
import Tyr.EventSkeleton.Examples.RimlessWheel
import Tyr.EventSkeleton.Examples.CompassGait
import Tyr.EventSkeleton.Examples.Rod2D
import Tyr.EventSkeleton.Examples.FourBar
import Tyr.EventSkeleton.Examples.MassSpringCloth
import Tyr.EventSkeleton.Examples.Deformable
import Tyr.EventSkeleton.Examples.HydroelasticDemos
import Tyr.EventSkeleton.Examples.InclinedPlaneBody
import Tyr.EventSkeleton.Examples.HydroelasticBallPlate
import Tyr.EventSkeleton.Examples.Fibonacci
import Tyr.EventSkeleton.Examples.SimpleSystems
import Tyr.EventSkeleton.Examples.VanDerPol
import Tyr.EventSkeleton.Examples.SimpleGripper
import Tyr.EventSkeleton.Examples.PlanarGripper
import Tyr.EventSkeleton.Examples.KukaIiwaArm
import Tyr.EventSkeleton.Examples.KinovaJacoArm
import Tyr.EventSkeleton.Examples.AllegroHand
import Tyr.EventSkeleton.Examples.SceneGraph
import Tyr.EventSkeleton.Examples.HardwareSim
import Tyr.EventSkeleton.Examples.Zmp
import Tyr.EventSkeleton.Examples.Atlas
import Tyr.EventSkeleton.Examples.CubicPolynomial
import Tyr.EventSkeleton.Examples.Strandbeest

/-!
# Tyr.EventSkeleton

Separate event-skeleton differentiation surface for hybrid, stochastic, and
branching systems.  This is intentionally independent from `Tyr.AD.Elim`; a
future adapter can map `localSchurBlock` moves onto the existing AlphaGrad
vertex eliminator.
-/
