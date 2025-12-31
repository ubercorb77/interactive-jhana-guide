# function spec: pose-invariant smile ratio from mediapipe face mesh

## goal

compute a scalar quantity

[
R = \frac{\text{smile length}}{\text{static facial reference length}}
]

that is:

* invariant to head **translation**
* invariant to head **rotation** (yaw, pitch, roll)
* invariant to **scale / distance to camera**
* stable under normal facial motion

the input is a **single frame** of mediapipe face mesh landmarks.

---

## inputs

* a set of (N = 468) face landmarks from mediapipe

* each landmark is a 3-vector:
  [
  \mathbf{p}_i = (x_i, y_i, z_i)
  ]
  where:

  * (x_i, y_i) are normalized image coordinates
  * (z_i) is relative depth (negative = closer to camera)

* predefined landmark indices for:

  * left mouth corner
  * right mouth corner
  * left eye anchor
  * right eye anchor
  * nose tip or face center anchor

(these indices already exist in the codebase or mediapipe docs)

---

## output

* a single floating-point value (R), dimensionless
* optionally smoothed temporally by the caller

---

## mathematical overview (high level)

the computation has **four conceptual stages**:

1. interpret mediapipe landmarks as a 3d point cloud
2. remove global translation
3. remove global rotation by defining a canonical face coordinate frame
4. measure distances in this canonical frame and take a ratio

the key idea is:

> **you do not measure distances in camera space; you measure distances in face space.**

---

## stage 1: treat landmarks as 3d points

even though mediapipe is monocular, the output landmarks form a **consistent 3d embedding of the face**.

we assume:
[
\mathbf{p}_i \in \mathbb{R}^3
]

no projection math is needed here — just work directly in 3d.

---

## stage 2: remove translation

### motivation

absolute position of the face in the camera frame is irrelevant. all distances should be measured relative to the face itself.

### procedure

choose a **stable anchor landmark**, typically:

* nose tip
* midpoint between eyes
* face mesh center

call this point:
[
\mathbf{p}_\text{anchor}
]

translate all points:
[
\mathbf{p}'_i = \mathbf{p}*i - \mathbf{p}*\text{anchor}
]

after this:

* the anchor lies at the origin
* all remaining variation is shape + orientation

---

## stage 3: remove rotation (core geometric step)

this is the most important part.

### idea

define a **local coordinate system attached to the face**, using anatomical landmarks.

once all faces are expressed in this coordinate system:

* head rotation disappears
* distances become comparable frame-to-frame

---

### 3.1 define face axes

#### x-axis (left–right direction)

use eye landmarks:

[
\mathbf{v}*x = \mathbf{p}'*{\text{right eye}} - \mathbf{p}'_{\text{left eye}}
]

normalize:
[
\hat{\mathbf{x}} = \frac{\mathbf{v}_x}{|\mathbf{v}_x|}
]

this axis points horizontally across the face.

---

#### y-axis (up–down direction)

use chin and forehead (or nose bridge):

[
\mathbf{v}*y = \mathbf{p}'*{\text{forehead}} - \mathbf{p}'_{\text{chin}}
]

normalize:
[
\hat{\mathbf{y}} = \frac{\mathbf{v}_y}{|\mathbf{v}_y|}
]

this axis points vertically along the face.

---

#### z-axis (face normal)

compute the cross product:

[
\hat{\mathbf{z}} = \hat{\mathbf{x}} \times \hat{\mathbf{y}}
]

then renormalize:
[
\hat{\mathbf{z}} \leftarrow \frac{\hat{\mathbf{z}}}{|\hat{\mathbf{z}}|}
]

(optional but recommended: re-orthogonalize (\hat{\mathbf{y}}) as
(\hat{\mathbf{y}} = \hat{\mathbf{z}} \times \hat{\mathbf{x}}))

---

### 3.2 build rotation matrix

construct a rotation matrix whose columns are the face axes:

[
R =
\begin{bmatrix}
\hat{\mathbf{x}} & \hat{\mathbf{y}} & \hat{\mathbf{z}}
\end{bmatrix}
]

this matrix maps **face coordinates → camera coordinates**.

to rotate points *into face space*, apply the transpose:

[
\mathbf{q}_i = R^\top \mathbf{p}'_i
]

now:

* all faces are aligned
* head pose is removed
* remaining variation is facial expression

---

## stage 4: measure smile and reference lengths

all distances are now computed using (\mathbf{q}_i).

---

### 4.1 smile length

use mouth corner landmarks:

[
L_{\text{smile}} =
\left|
\mathbf{q}*{\text{right mouth}} -
\mathbf{q}*{\text{left mouth}}
\right|
]

this measures true mouth width in face space, unaffected by yaw.

---

### 4.2 static reference length

choose a facial length that:

* is minimally affected by smiling
* is stable across frames

recommended options:

* inter-ocular distance
* eye corner to eye corner
* nose bridge length

example:

[
L_{\text{ref}} =
\left|
\mathbf{q}*{\text{right eye}} -
\mathbf{q}*{\text{left eye}}
\right|
]

---

### 4.3 ratio

compute final output:

[
R = \frac{L_{\text{smile}}}{L_{\text{ref}}}
]

this ratio is:

* dimensionless
* scale-invariant
* robust to distance from camera
* comparable across frames and (to some extent) individuals

---

## assumptions and limitations (important to document)

* mediapipe z is **relative**, not metric
  → ratios are valid; absolute distances are not

* extreme head poses (> ~45° yaw) reduce accuracy

* reference landmarks must be chosen to minimize expression coupling

* lighting and occlusion can degrade landmark quality

---

## integration notes (for modifying an existing function)

* this logic replaces any computation based purely on 2d pixel distances
* translation removal and rotation normalization should happen **inside the function**
* the caller should treat the output as a normalized expression signal
* optional temporal smoothing (EMA, Kalman) should be done externally

---

## one-sentence summary for the code comment

> “this function computes a pose-invariant smile metric by rotating mediapipe’s 3d face landmarks into a canonical face coordinate frame and measuring mouth width relative to a stable facial reference.”