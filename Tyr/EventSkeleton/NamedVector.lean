namespace Tyr.EventSkeleton

/-!
# Drake-style named vector boundaries

Several Drake examples still expose hand-written descendants of historically
generated `BasicVector` classes.  The event-skeleton port records those files as
named-vector boundaries: coordinate order, defaults, bounds, and source paths.
-/

structure NamedVectorBoundary where
  typeName : String
  headerPath : String
  implementationPath? : Option String := none
  coordinateNames : Array String
  defaults : Array Float
  lowerBounds : Array (Option Float)
  upperBounds : Array (Option Float)
  movedFromAccessThrows : Bool := true
  supportsNamedVariables : Bool := true
  deriving Repr, Inhabited

private def hasDuplicateString (xs : Array String) : Bool := Id.run do
  let mut duplicate := false
  for i in [:xs.size] do
    for j in [:(xs.size - i - 1)] do
      let k := i + j + 1
      if xs[i]! == xs[k]! then
        duplicate := true
  return duplicate

namespace NamedVectorBoundary

def dimension (boundary : NamedVectorBoundary) : Nat :=
  boundary.coordinateNames.size

def hasCoordinate (boundary : NamedVectorBoundary) (name : String) : Bool :=
  boundary.coordinateNames.contains name

def indexOf? (boundary : NamedVectorBoundary) (name : String) : Option Nat :=
  boundary.coordinateNames.findIdx? (fun candidate => candidate == name)

def validate? (boundary : NamedVectorBoundary) : Except String Unit := do
  if boundary.typeName.isEmpty then
    .error "named vector boundary requires a type name"
  if boundary.headerPath.isEmpty then
    .error s!"{boundary.typeName} must record its Drake header path"
  if boundary.coordinateNames.isEmpty then
    .error s!"{boundary.typeName} must expose at least one coordinate"
  if hasDuplicateString boundary.coordinateNames then
    .error s!"{boundary.typeName} coordinate names must be unique"
  if boundary.defaults.size != boundary.dimension then
    .error s!"{boundary.typeName} defaults have size {boundary.defaults.size}, expected {boundary.dimension}"
  if boundary.lowerBounds.size != boundary.dimension then
    .error s!"{boundary.typeName} lower bounds have size {boundary.lowerBounds.size}, expected {boundary.dimension}"
  if boundary.upperBounds.size != boundary.dimension then
    .error s!"{boundary.typeName} upper bounds have size {boundary.upperBounds.size}, expected {boundary.dimension}"
  for i in [:boundary.dimension] do
    let value := boundary.defaults[i]!
    if !value.isFinite then
      .error s!"{boundary.typeName}.{boundary.coordinateNames[i]!} default is not finite: {value}"
    match boundary.lowerBounds[i]!, boundary.upperBounds[i]! with
    | some lo, some hi =>
        if lo > hi then
          .error s!"{boundary.typeName}.{boundary.coordinateNames[i]!} has inverted bounds [{lo}, {hi}]"
        if value < lo || value > hi then
          .error s!"{boundary.typeName}.{boundary.coordinateNames[i]!} default {value} is outside [{lo}, {hi}]"
    | some lo, none =>
        if value < lo then
          .error s!"{boundary.typeName}.{boundary.coordinateNames[i]!} default {value} is below lower bound {lo}"
    | none, some hi =>
        if value > hi then
          .error s!"{boundary.typeName}.{boundary.coordinateNames[i]!} default {value} is above upper bound {hi}"
    | none, none => pure ()

end NamedVectorBoundary

end Tyr.EventSkeleton
