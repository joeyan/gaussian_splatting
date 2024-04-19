# Gaussian Splatting Algorithm Details

## Representation

A gaussian is represented by the following parameters:

|                                 | Variable Name | N Params     | Valid Range | Activation      |
|---------------------------------|---------------|--------------|-------------|-----------------|
| Mean Position                   | xyz           | 3            | (-inf, inf) | none            |
| Color                           | rgb           | 3            | [0, 1]      | none or sigmoid |
| Opacity                         | opacity       | 1            | [0, 1]      | sigmoid         |
| Scale                           | scale         | 3            | (0, inf)    | exponential     |
| Spherical Harmonic Coefficients | sh            | 0, 9, 24, 45 |             | none            |


## Forward Pass

#### 1 Dimensional Gaussian Distribution

Let's start with 1 dimensional gaussian probability density function where $\sigma$ is the standard deviation and $\mu$ is the mean.

$$ g(x) = \frac{1}{\sigma\sqrt{2\pi}}exp\left(\frac{-(x - \mu)}{2\sigma^2}\right)$$ 


![image](https://github.com/joeyan/gaussian_splatting/assets/17635504/105862db-3c2e-4579-bcc9-1ac4e9b8da44)


The peak probability changes with $\sigma$ due to the normalization term $\frac{1}{\sigma\sqrt{2\pi}}$. For gaussian splatting purposes, this means that the opacity of the gaussian is dependent on the size of the gaussian. Large gaussians will require a very high opacity factor to make them visible and small gaussians may oversaturate the image. Additionally, this makes it difficult to compare opacity across different sized gaussians which is required for the opacity reset and delete in the adaptive control algorithm. 


Dropping the normalization term decouples the density from the opacity of the gaussian. 

$$ g(x) = exp\left(\frac{-(x - \mu)}{2\sigma^2}\right)$$ 

![image](https://github.com/joeyan/gaussian_splatting/assets/17635504/be238e84-8baa-40f8-84e2-fcddf70f6391)

#### Multivariate Gaussian Distribution

The generalized multivariate gaussian distribution _without normalization_ can be defined as:

$$ g(\boldsymbol{x}) = exp\left( - \frac{1}{2}(\boldsymbol{x} - \boldsymbol{\mu})^T \Sigma^{-1}(\boldsymbol{x} - \boldsymbol{\mu})\right) $$

Where $\boldsymbol{\sigma}$ and $\boldsymbol{\mu}$ are the vector standard deviation and means and $\Sigma$ is the covariance matrix. The covariance matrix is a symmetric and positive semi-definite $NxN$ matrix where N is the number of dimensions in the distribution.


#### Optimizing 3D Gaussians

The 3D gaussians are represented by a $3x3$ covariance matrix. The symmetry of the matrix can be maintained by only optimizing the 6 parameters in the upper triangular portion of the matrix but it is much more difficult to constrain the matrix to be positive semi-definite. Instead of optimizing the 3D covariance matrix directly, the authors construct the matrix representation of an ellipsoid from 3 scale terms and a 3D rotation. This process is effectively an inverse Principal Component Analysis.

The covariance matrix can be decomposed into its eigenvalues and eigenvectors:
$$\Sigma\boldsymbol{v}=\lambda\boldsymbol{v}$$ 

Where $\boldsymbol{v}$ is an eigenvector of $\Sigma$ and $\lambda$ is the corresponding eigenvalue. This creates the following system of equations:

$$\Sigma\begin{bmatrix} \boldsymbol{v_0} & \boldsymbol{v_1} & \boldsymbol{v_2}  \end{bmatrix}=\begin{bmatrix} \boldsymbol{v_0} & \boldsymbol{v_1} & \boldsymbol{v_2}\end{bmatrix} \begin{bmatrix} \lambda_1 & 0 & 0 \\\ 0 & \lambda_2 & 0 \\\ 0 & 0 & \lambda_3  \end{bmatrix}$$

In matrix form where the columns of $V$ are the eigenvectors:
$$\Sigma V = VL$$ 
 Rearranging yields:
$$\Sigma = VLV^{-1}$$ 

In PCA, the eigenvectors define the direction of largest variance and the eigenvalues define magnitude of the variance. The eigenvector matrix is equivalent to the rotation matrix of the axes of largest variance to the starting reference frame. Rotation matrices/eigenvectors are orthogonal so the inverse is equal to the transpose.

$$ \Sigma = RLR^T = R\begin{bmatrix} \lambda_1 & 0 & 0 \\\ 0 & \lambda_2 & 0 \\\ 0 & 0 & \lambda_3  \end{bmatrix} R^T$$

Defining the scale matrix with $s_n=\sqrt{\lambda_n}$: 

$$S = \begin{bmatrix} s_{1} & 0 & 0 \\\ 0 & s_{2} & 0 \\\ 0 & 0 & s_{3} \end{bmatrix}$$

Substituting back in: 

$$\Sigma = RSS^TR^T = RS(RS)^T$$ 

The $3x3$ rotation matrix can be represented with a quaternion leaving 7 total parameters to be optimized: $q_w, q_x, q_y, q_z$ and $s_1, s_2, s_3$. The resulting covariance matrix is guaranteed to be positive semi-definite since  $\lambda_n = s_{n}^{2}$ and therefore all eigenvalues are non-negative.


#### Projecting 3D Gaussians

The 3D covariance matrix into a 2D covariance matrix by:

$$\Sigma_2 = JW\Sigma_3W^TJ^T$$

Where W is the $3x3$ rotation matrix representing the viewing transform and J is the $2x3$ "Jacobian of the affine approximation of the projective transformation". With large focal lengths and small gaussians, the approximation should work well. 

$$J = \begin{bmatrix} f_x / z & 0 & -f_x x/z^2 \\\ 0 & f_y/z & -f_yy/z^2\end{bmatrix}$$

Note: The original [_EWA Splatting_](https://www.cs.umd.edu/~zwicker/publications/EWASplatting-TVCG02.pdf) paper (equation 34) uses a projection plane at $z=1$ which is equivalent to $f_x = f_y = 1$

#### Creating Images with 2D Gaussians

The mean of the 2D gaussian can be calculated by the standard pinhole projection of the 3D mean:

$$\begin{bmatrix} \bar{u} & \bar{v} \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\\ 0 & f_y & c_y \end{bmatrix} \begin{bmatrix} \bar{x}/\bar{z} \\\ \bar{y}/\bar{z} \\\ 1\end{bmatrix}$$ 

The unnormalized probability density of each pixel can be computed:

$$g(\boldsymbol{x}) = exp\left( - \frac{1}{2}(\boldsymbol{x} - \boldsymbol{\mu})^T \Sigma^{-1}(\boldsymbol{x} - \boldsymbol{\mu})\right)$$

with

$$\boldsymbol{x} - \boldsymbol{\mu} = \begin{bmatrix} \bar{u} - u \\\ \bar{v} -v \end{bmatrix}$$

and 

$$\Sigma^{-1} = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a\end{bmatrix}$$

Note: in the CUDA implementation, the matrix representation of the conic is used instead of a 2D covariance matrix. The 2D covariance matrix is symmetric and only has 3 degrees of freedom. Since $b=c$ is always true, the 2D covariance matrix can be compactly stored as 3 variables $a, b, c$. 


#### Tile Based Rasterization

Evaluating the probability of every gaussian at every pixel in the image would be too slow for real-time rendering. Luckily, most gaussians only cover a small portion of the rendered image in order to accurately reconstruct the fine detail in the scene. The authors choose to break the image down in to 16 by 16 pixel tiles and only render the gaussians that have _significant contribution_ for each tile. 

Currently, the output images are converted to `uint8` and thus only have a resolution of $1/255 \approx 0.00392$ which is very close to probability of a gaussian distribution at $3\sigma$. The mapping of gaussian to tiles is computed by finding the intersection of the gaussian distribution at $3\sigma$ and the tiles.

An oriented bounding box of the ellipse at $3\sigma$ can be computed from the 2D covariance matrix:

$$\Sigma_2 = \begin{bmatrix} a & b \\\ c & d \end{bmatrix}$$ 

First, compute the two eigenvalues for a 2x2 symmetric matrix:

$$ \lambda_1 = \frac{a + d + \sqrt{(a-d)^2 + 4bc}}{2} $$ 
and
$$ \lambda_2 = \frac{a + d - \sqrt{(a-d)^2 + 4bc}}{2} $$ 

The radii at $3\sigma$ is:

$$r_1 = 3\sqrt{\lambda_1}$$ 

$$r_2 = 3\sqrt{\lambda_2}$$ 

The orientation can be computed with:

If $b \neq 0$:

$$\theta = arctan\left(\frac{\lambda_1}{b}\right)$$

if $b = 0$ and $a >= d$

$$\theta = 0$$ 

if $b = 0$ and $a < d$

$$\theta = \frac{\pi}{2}$$ 


The four corner points of the oriented bounding box can be constructed from $r_1, r_2, \theta$ by initializing the 4 corner points in the axis-aligned bounding box and rotating them with the 2D rotation matrix:

$$R = \begin{bmatrix} cos\theta & - sin\theta \\\ sin\theta & cos\theta \end{bmatrix}$$

The intersection between the oriented bounding box and the tile can be computed by using the Separating Axis Theorem. In this case, the SAT can be simplified since each bounding box has two sets of parallel axes - only one of the two need to be checked. Additionally, the axes of the tile are axis-aligned eliminating two projection steps. 

Here is a really good deep dive on the [Separating Axis Theorem](https://dyn4j.org/2010/01/sat/). 

This example image shows 3 gaussians splatted on the image. The green boxes are the oriented bounding boxes at $3\sigma$ and the tiles that intersect are colored in white. 

![image](https://github.com/joeyan/gaussian_splatting/assets/17635504/741a17f8-3de0-4561-bc64-309f5a38c1cd)

#### Alpha Compositing
The RGB value of each pixel is computed by $\alpha$ blending the gaussians from front-to-back. The rgb values of each pixel can be computed with:

$$C(u, v) = \sum_{i=1}^{N} \alpha_{i}c_{i}w_{i}$$

$$ w_{i} =  (1 - \sum_{j=0}^{i-1}\alpha_{j}w_j )$$

$$ w_0 = 1 $$

$$ \alpha_{i} = o_i g_{i}(u, v)$$
 
Where $c_{i}$ and $o_i$ are the color and opacity of the $i^{th}$ gaussian and $g_{i}(u, v)$ is the probabilty of the the $i^{th}$ gaussian at the pixel coordinates $u, v$. 


## Backward Pass
For implemented forward/backwards passes in PyTorch - see analytic_diff.ipynb

#### Notation
For simplicity, all gradients are denoted by $\nabla$ 


#### Camera Projection
Computing the reverse-mode derivatives for the camera projection is fairly straightforward. For the forward projection:

$$\begin{bmatrix} u \\\ v \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\\ 0 & f_y & c_y \end{bmatrix} \begin{bmatrix} \frac{x}{z} \\\ \frac{y}{z} \\\ 1 \end{bmatrix}$$

The Jacobian is (same Jacobian that is used to project the 3D gaussians to 2D): 

$$J = \begin{bmatrix} f_x / z & 0 & -f_x x/z^2 \\\ 0 & f_y/z & -f_yy/z^2\end{bmatrix}$$

Computing the vector-Jacobian product yields:

$$\begin{bmatrix} \nabla{x} & \nabla{y} & \nabla{z} \end{bmatrix} = \begin{bmatrix}\nabla{u} & \nabla{v}\end{bmatrix} \begin{bmatrix} f_x / z & 0 & -f_x x/z^2 \\\ 0 & f_y/z & -f_yy/z^2\end{bmatrix}$$ 


#### Normalizing Quaternion
For a quaternion $q$:

$$ q = \begin{bmatrix} w & x & y & z\end{bmatrix}$$

The normalized quaternion can be computed with:

$$ \hat{q} = \begin{bmatrix} \frac{w}{\lVert q \rVert} & \frac{x}{\lVert q \rVert} & \frac{y}{\lVert q \rVert} & \frac{z}{\lVert q \rVert}\end{bmatrix}$$

The vector-Jacobian products are:


$$ {\nabla w} = \nabla{\hat{q}}^{T}\begin{bmatrix}- \frac{w^{2}}{\left(w^{2} + x^{2} + y^{2} + z^{2}\right)^{\frac{3}{2}}} + \frac{1}{\sqrt{w^{2} + x^{2} + y^{2} + z^{2}}} \\\ - \frac{w x}{\left(w^{2} + x^{2} + y^{2} + z^{2}\right)^{\frac{3}{2}}} \\\ -\frac{w y}{\left(w^{2} + x^{2} + y^{2} + z^{2}\right)^{\frac{3}{2}}} \\\ - \frac{w z}{\left(w^{2} + x^{2} + y^{2} + z^{2}\right)^{\frac{3}{2}}}\end{bmatrix} $$

$$ \nabla x = \nabla{\hat{q}}^{T}\begin{bmatrix}- \frac{w x}{\left(w^{2} + x^{2} + y^{2} + z^{2}\right)^{\frac{3}{2}}} \\\ -\frac{x^{2}}{\left(w^{2} + x^{2} + y^{2} + z^{2}\right)^{\frac{3}{2}}} + \frac{1}{\sqrt{w^{2} + x^{2} + y^{2} + z^{2}}} \\\ -\frac{x y}{\left(w^{2} + x^{2} + y^{2} + z^{2}\right)^{\frac{3}{2}}} \\\ -\frac{x z}{\left(w^{2} + x^{2} + y^{2} + z^{2}\right)^{\frac{3}{2}}}\end{bmatrix} $$

$$ \nabla y = \nabla{\hat{q}}^{T}\begin{bmatrix}- \frac{w y}{\left(w^{2} + x^{2} + y^{2} + z^{2}\right)^{\frac{3}{2}}} \\\ -\frac{x y}{\left(w^{2} + x^{2} + y^{2} + z^{2}\right)^{\frac{3}{2}}} \\\ -\frac{y^{2}}{\left(w^{2} + x^{2} + y^{2} + z^{2}\right)^{\frac{3}{2}}} + \frac{1}{\sqrt{w^{2} + x^{2} + y^{2} + z^{2}}} \\\ -\frac{y z}{\left(w^{2} + x^{2} + y^{2} + z^{2}\right)^{\frac{3}{2}}}\end{bmatrix} $$ 

$$ \nabla z = \nabla{\hat{q}}^{T}\begin{bmatrix}- \frac{w z}{\left(w^{2} + x^{2} + y^{2} + z^{2}\right)^{\frac{3}{2}}} \\\ -\frac{x z}{\left(w^{2} + x^{2} + y^{2} + z^{2}\right)^{\frac{3}{2}}} \\\ -\frac{y z}{\left(w^{2} + x^{2} + y^{2} + z^{2}\right)^{\frac{3}{2}}} \\\ -\frac{z^{2}}{\left(w^{2} + x^{2} + y^{2} + z^{2}\right)^{\frac{3}{2}}} + \frac{1}{\sqrt{w^{2} + x^{2} + y^{2} + z^{2}}}\end{bmatrix} $$



#### Normalized Quaternion to Rotation Matrix

For a normalized quaternion $\hat{q}$ (Note: the notation is a little sloppy here. In this section $w$, $x$, $y$, and $z$ are the components of $\hat{q}$ not $q$ unlike the previous section):

$$R = \begin{bmatrix}- 2 y^{2} - 2 z^{2} + 1 & - 2 w z + 2 x y & 2 w y + 2 x z \\\ 2 w z + 2 x y & - 2 x^{2} - 2 z^{2} + 1 & - 2 w x + 2 y z \\\ -2 w y + 2 x z & 2 w x + 2 y z & - 2 x^{2} - 2 y^{2} + 1\end{bmatrix} $$

Computing the Jacobians and computing the vector-Jacobian product:

$$  \nabla{w}= \nabla{R}^T\begin{bmatrix}0 & - 2 z & 2 y \\\ 2 z & 0 & - 2 x \\\ -2 y & 2 x & 0\end{bmatrix} $$

$$  \nabla{x} = \nabla{R}^T\begin{bmatrix}0 & 2 y & 2 z \\\ 2 y & - 4 x & - 2 w \\\ 2 z & 2 w & - 4 x\end{bmatrix} $$ 

$$  \nabla{y} = \nabla{R}^T \begin{bmatrix}- 4 y & 2 x & 2 w \\\ 2 x & 0 & 2 z \\\ -2 w & 2 z & - 4 y\end{bmatrix} $$ 

$$  \nabla{z} = \nabla{R}^T \begin{bmatrix}- 4 z & - 2 w & 2 x \\\ 2 w & - 4 z & 2 y \\\ 2 x & 2 y & 0\end{bmatrix} $$



#### 3D Covariance Matrix
The reverse mode differentiation for matrix operations is documented in [An extended collection of matrix derivative results for forward and reverse mode algorithmic differentiation](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf). 

For matrix multiplication in section 2.2.2:

$$C = AB$$

The gradients can be computed by:

$$\nabla{A} = \nabla{C}B^{T}$$ 

$$\nabla{B} = A^{T}\nabla{C} $$ 

This can be applied to the computation of the 3D covariance matrix. 

$$\Sigma_{3D} = RS(RS)^{T}$$

Breaking this down into two matrix multiplication operations:

$$M = RS$$

$$ \Sigma_{3D} = MM^{T} $$

The gradients can be computed by:

$$\nabla{M} = \nabla{\Sigma_{3D}}(M^{T})^{T} = \nabla{\Sigma_{3D}}M$$

$$\nabla{M^{T}} = M^{T} \nabla{\Sigma_{3D}}$$

Computing the gradients of the components of $M$:

$$ \nabla{R} = \nabla{M} S^{T}$$

$$ \nabla{S} = R^{T} \nabla{M}$$

Computing the gradients of the components of $M^{T}$:

$$\nabla{R^{T}} = S\nabla{M^{T}}$$

$$\nabla{S^{T}} = \nabla{M^{T}} R$$

Combining the gradients of the transposed components:

$$\nabla{R} = \nabla{M} S^{T} + (S\nabla{M^{T}})^T = \nabla{M} S^{T} + (\nabla{M^{T}})^{T}S^{T} $$

$$\nabla{S} = R^{T}\nabla{M} + (\nabla{M}^{T}R)^{T} = R^{T}\nabla{M} + R^{T}(\nabla{M^{T}})^{T} $$

Substituting in for $M$:

$$\nabla{R} = \nabla{\Sigma_{3D}}RSS^{T} + (\nabla{\Sigma_{3D}})^{T}RSS^{T}$$ 

$$\nabla{S} = R^{T}\nabla{\Sigma_{3D}}RS + R^{T}(\nabla{\Sigma_{3D}})^{T}RS$$

The expression can be further simplified since $S$, $\Sigma_{3D}$, and $\nabla{\Sigma_{3D}}$ are all symmetric:
$$\nabla{R} = 2\nabla{\Sigma_{3D}}RSS$$

$$\nabla{S} = 2R^{T}\nabla{\Sigma_{3D}}RS$$


#### 2D Covariance Matrix/Conic
Using the First Quadratic Form from _An extended collection of matrix derivative results_ in section 2.3.2: 

$$C=B^{T}AB$$

$$\nabla{A} = B\nabla{C}B^{T}$$
$$\nabla{B} = AB(\nabla{C})^{T} + A^{T}B\nabla{C}$$

Subsitituting in the 2D covariance matrix calculation: 

$$\Sigma_{2D} = JW\Sigma{3D}(JW)^T$$

$$A = \Sigma_{3D} $$

$$B = (JW)^T $$


$$\nabla{\Sigma_{3D}} = (JW)^{T}\nabla{\Sigma_{2D}}JW$$


$$\nabla{(JW)^{T}} = \Sigma_{3D}(JW)^{T}(\nabla{\Sigma_{2D}})^{T} + \Sigma_{3D}^{T}(JW)^{T} \nabla{\Sigma_{2D}}$$ 

With symmetric $\Sigma_{3D}$ and $\nabla{\Sigma_{2D}}$:

$$ \nabla{(JW)^{T}} = 2\Sigma_{3D}(JW)^{T} \nabla{\Sigma_{2D}}$$

Computing the gradient with respect to $J$ and $W$:

$$(JW)^{T} = W^TJ^T$$

$$C = (JW)^{T}$$

$$A = W^T$$ 

$$B = J^T$$

$$\nabla{W^T} = \nabla{(JW)^{T}} (J^T)^T = \nabla{(JW)^{T}}J$$

$$\nabla{J^T} = (W^T)^T\nabla{(JW)^{T}} = W\nabla{(JW)^{T}} $$


Transposing and substituting back in:

$$ \nabla{W} = (2\Sigma_{3D}(JW)^{T} \nabla{\Sigma_{2D}}J)^T = 2J^T\nabla{\Sigma_{2D}}JW\Sigma_{3D}$$

$$ \nabla{J} = (2W\Sigma_{3D}(JW)^{T} \nabla{\Sigma_{2D}})^T = 2\nabla{\Sigma_{2D}}JW\Sigma_{3D}W^T$$ 


#### Evaluating the Gaussian
The unnormalized probability of the Gaussian function:


$$ g(\boldsymbol{x}) = exp\left( - \frac{1}{2}(\boldsymbol{x} - \boldsymbol{\mu})^T \Sigma^{-1}(\boldsymbol{x} - \boldsymbol{\mu})\right) $$

With the Mahalanobis distance:

$$ d_M = \sqrt{(\boldsymbol{x} - \boldsymbol{\mu})^T \Sigma^{-1}(\boldsymbol{x} - \boldsymbol{\mu})} $$

$$ g(\boldsymbol{x}) = exp\left( - \frac{1}{2} d_M^2\right) $$


Evaluated at the pixel $(u, v)$ and using the conic representation of the 2D covariance matrix:

$$ d_M^2 = \begin{bmatrix} u - \mu_u \\\ v - \mu_v \end{bmatrix}^T \begin{bmatrix} a & b \\\ b & c \end{bmatrix}^{-1}\begin{bmatrix} u - \mu_u \\\ v - \mu_v \end{bmatrix}$$

$$ d_M^2 = \frac{1}{(ac - 2b)}\begin{bmatrix} u - \mu_u \\\ v - \mu_v \end{bmatrix}^T \begin{bmatrix} c & -b \\\ -b & a \end{bmatrix}\begin{bmatrix} u - \mu_u \\\ v - \mu_v \end{bmatrix}$$

Simplifying with:

$$ \Delta u = u - \mu_u $$

$$ \Delta v = v - \mu_v $$

$$ d_M^2 = \frac{a \Delta v^{2} - 2b \Delta u \Delta v + c \Delta u^{2}}{(ac - 2b)}$$


Computing vector-Jacobian products:

$$ \nabla{\Sigma_{2D}} = \nabla{d_M^2} \begin{bmatrix} \frac{\partial{d_M^2}}{\partial{a}} & \frac{\partial{d_M^2}}{\partial{b}} \\\ \frac{\partial{d_M^2}}{\partial{b}} & \frac{\partial{d_M^2}}{\partial{c}}  \end{bmatrix}$$

Since the covariance matrix is symmetric, the gradient for $b$ is the sum of the two (identical) components:

$$ \nabla{a} =  \nabla{d_M^2}\left(- \frac{c \left(a v^{2} - 2 b u v + c u^{2}\right)}{\left(a c - 2 b\right)^{2}} + \frac{v^{2}}{a c - 2 b} \right)$$ 

$$ \nabla{b} =  2 \nabla{d_M^2}\left( - \frac{u v}{a c - 2 b} + \frac{\left(a v^{2} - b u v + c u^{2}\right)}{\left(a c - 2 b\right)^{2}} \right)$$

$$ \nabla{c} = - \nabla{d_M^2}\left(\frac{a \left(a v^{2} - 2 b u v + c u^{2}\right)}{\left(a c - 2 b\right)^{2}} + \frac{u^{2}}{a c - 2 b} \right)  $$ 




The gradient of $d_M^2$ is computed from the the probability density at $(u, v)$:

$$ \nabla{d_M^2} = - \frac{1}{2}\nabla{g(u, v)}g(u, v) = -\frac{1}{2}\nabla{g(u, v)}exp\left( - \frac{1}{2} d_M^2\right) $$ 


#### Alpha Compositing
For the first compositing step: 

$$ C_{image} = C_{0} \alpha_{0} + C_{1}\alpha_{1}(1 - \alpha_{0}) $$

The gradients for the gaussian colors:

$$ \nabla{C_0} = \nabla{C_{image}} \alpha_{0} $$

$$ \nabla{C_1} = \nabla{C_{image}} \alpha_{1} (1 - \alpha_{0}) $$

The gradients for $\alpha$: 

$$ \nabla{\alpha_{0}} = \nabla{C_{image}} (C_{0} - C_{1}\alpha_{1}) $$ 

$$ \nabla{\alpha_{1}} = \nabla{C_{image}} C_{1} (1 - \alpha_{0})   $$

Now adding in a third gaussian:

$$ C_{image} = C_{0} \alpha_{0} + C_{1}\alpha_{1}(1 - \alpha_{0}) + C_{2}\alpha_{2}(1 - (\alpha_{0} + \alpha_{1}(1 - \alpha_{0}))) $$

The color gradients for the first two gaussians remain the same. For the third gaussian:

$$ \nabla{C_{2}} = \nabla{C_{image}}\alpha_{2}(1 - (\alpha_{0} + \alpha_{1}(1 - \alpha_{0})) $$

The gradients change for all three alpha values:

$$ \nabla{\alpha_{0}} = \nabla{C_{image}} (C_{0} - C_{1}\alpha_{1} - C_{2}\alpha_{2}(1 + \alpha_{1}) ) $$ 

$$ \nabla{\alpha_{1}} = \nabla{C_{image}} (C_{1} (1 - \alpha_{0}) - C_{2} \alpha_{2}(1 - \alpha_{0}))  $$

$$ \nabla{\alpha_{2}} = \nabla{C_{image}} C_{2} (1 - (\alpha_{0} + \alpha_{1}(1 - \alpha_{0})) $$

Using the patterns in the above steps, the gradients for compositing N gaussians can be computed. For the color gradient, this is straightforward when computing $\nabla{C_{n}}$ iteratively starting from $n=0$:

$$ \nabla{C_{n}} = \nabla{C_{image}} \alpha_{n} w_{n} $$

$$ w_{n} =  (1 - \sum_{i=0}^{n-1}\alpha_{i}w_i )$$

$$ w_0 = 1 $$

The $\alpha$ gradients are trickier as $\nabla{\alpha_0}$ and $\nabla{\alpha_n}$ are both dependent on all $\alpha$ values. The final $\alpha$ value is the easiest to calculate:

$$ \nabla{\alpha_n} = \nabla{C_{image}} C_{n} w_n $$

The final weight can also be easily saved at the end of the forward pass. It is also possible to compute the previous/next weight:

$$ w_n = 1 - w_{n-1} $$

$$ w_{n+1} = 1 - (w_{n-1} + \alpha_{n} w_n) = 1 - w_{n-1} - \alpha_{n}w_{n} $$

Substituting in $w_{n}$ and regrouping:

$$ w_{n+1} = 1 - w_{n-1} - \alpha_{n}(1 - w_{n-1}) = 1 - w_{n-1} - \alpha_n + \alpha_n w_{n-1} = (1- \alpha_n)(1-w_{n-1})$$

Substituting $w_{n}$ back out:

$$ w_{n+1} = (1 - \alpha_n) w_{n} $$

Revisiting the alpha compositing example:

$$ C_{image} = \sum_{i=0}^{n}C_i\alpha_{i}w_i $$ 

For any gaussian $m$, the gradient of $\alpha_m$ does not depend on any gaussians where $m > i$:

$$ \nabla{\alpha_m} = \nabla{C_{image}}\left(C_mw_m  +  \sum_{i=m+1}^{n}C_i\alpha_{i} \frac{\partial{w_i}}{\partial{\alpha_m}} \right)$$

Another way to think about $w_i$:

$$ w_i = (1 - \alpha_{i-1})(1 - \alpha_{i-2}) .... (1 - \alpha_0) $$

By multiplying out the $1-\alpha_m$ term:

$$ w_i = \frac{w_i}{1 - \alpha_m} + \frac{-\alpha_m w_i}{1 - \alpha_m}$$

The first term $\frac{w_i}{1 - \alpha_m}$ is not dependent on $\alpha_m$ so the partial derivative is:

$$ \frac{\partial{w_i}}{\alpha_m} = \frac{-w_i}{1 - \alpha_{m}} $$

Substituting back in:

$$ \nabla{\alpha_m} = \nabla{C_{image}}\left(C_mw_m  +  \sum_{i=m+1}^{n}C_i\alpha_{i} \frac{-w_i}{1 - \alpha_{m}} \right)$$

And simplifying:

$$ \nabla{\alpha_m} = \nabla{C_{image}}\left(C_mw_m  -  \frac{\sum_{i=m+1}^{n}C_i\alpha_{i} w_i }{1 - \alpha_{m}} \right)$$

The numerator of the fraction is now the accumulated color from the current gaussian to the last gaussian.

$$ \sum_{i=m+1}^{n}C_i\alpha_{i} w_i $$

By computing the gradients from back to front the weight and accumulated color can be efficiently computed by saving the final weight in the forward pass.

Starting with: 

$$ C_{accum} = 0$$

$$ w_i = w_{final} $$ 

At each iteration:

$$ \nabla{C_{i}} = \nabla{C_{image}} \alpha_{i} w_{i} $$

$$ \nabla{\alpha_i} = \nabla{C_{image}}\left(C_iw_i  -  \frac{C_{accum} }{1 - \alpha_{i}} \right)$$

Updating accumulated color and weight for the next step:

$$ C_{accum} = C_{accum} + C_i\alpha_{i} w_i$$

$$ w_{i - 1} = \frac{w_i}{1 -\alpha_i} $$
