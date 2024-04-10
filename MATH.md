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

![image](https://github.com/joeyan/gaussian_splatting/assets/17635504/741a17f8-3de0-4561-bc64-309f5a38c1cd)

#### Alpha Compositing
The RGB value of each pixel is computed by $\alpha$ blending the gaussians from front-to-back. The rgb values of each pixel can be computed with:

$$C(u, v) = \sum_{i=1}^{N} w_{i}c_{i}$$

$$ w_{i} = \alpha_i g_{i}(u, v) (1 - \sum_{j=0}^{i-1}w_{j})$$ 

Where $c_{i}$ and $\alpha_i$ are the color and opacity of the $i^{th}$ gaussian and $g_{i}(u, v)$ is the probabilty of the the $i^{th}$ gaussian at the pixel coordinates $u, v$. 


## Backward Pass
For implemented forward/backwards passes in PyTorch - see analytic_diff.ipynb

#### Notation
For simplicity, all gradients are denotied by $\partial$ 


#### Camera Projection
Computing the reverse-mode derivatives for the camera projection is fairly straightforward. For the forward projection:

$$\begin{bmatrix} u \\\ v \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\\ 0 & f_y & c_y \end{bmatrix} \begin{bmatrix} \frac{x}{z} \\\ \frac{y}{z} \\\ 1 \end{bmatrix}$$

The Jacobian is (same Jacobian that is used to project the 3D gaussians to 2D): 

$$J = \begin{bmatrix} f_x / z & 0 & -f_x x/z^2 \\\ 0 & f_y/z & -f_yy/z^2\end{bmatrix}$$

Computing the vector-Jacobian product yields:

$$\begin{bmatrix} \partial{x} & \partial{y} & \partial_{z} \end{bmatrix} = \begin{bmatrix}\partial{u} & \partial{v}\end{bmatrix} \begin{bmatrix} f_x / z & 0 & -f_x x/z^2 \\\ 0 & f_y/z & -f_yy/z^2\end{bmatrix}$$ 


#### Quaternion to Rotation Matrix




#### 3D Covariance Matrix
The reverse mode differentiation for matrix operations is documented in [An extended collection of matrix derivative results for forward and reverse mode algorithmic differentiation](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf). 

For matrix multiplication in section 2.2.2:

$$C = AB$$

The gradients can be computed by:

$$\partial{A} = \partial{C}B^{T}$$ 

$$\partial{B} = A^{T}\partial{C} $$ 

This can be applied to the computation of the 3D covariance matrix. 

$$\Sigma_{3D} = RS(RS)^{T}$$

Breaking this down into two matrix multiplication operations:

$$M = RS$$

$$ \Sigma_{3D} = MM^{T} $$

The gradients can be computed by:

$$\partial{M} = \partial{\Sigma_{3D}}(M^{T})^{T} = \partial{\Sigma_{3D}}M$$

$$\partial{M^{T}} = M^{T} \partial{\Sigma_{3D}}$$

Computing the gradients of the components of $M$:

$$ \partial{R} = \partial{M} S^{T}$$

$$ \partial{S} = R^{T} \partial{M}$$

Computing the gradients of the components of $M^{T}$:

$$\partial{R^{T}} = S\partial{M^{T}}$$

$$\partial{S^{T}} = \partial{M^{T}} R$$

Combining the gradients of the transposed components:

$$\partial{R} = \partial{M} S^{T} + (S\partial{M^{T}})^T = \partial{M} S^{T} + (\partial{M^{T}})^{T}S^{T} $$

$$\partial{S} = R^{T}\partial{M} + (\partial{M}^{T}R)^{T} = R^{T}\partial{M} + R^{T}(\partial{M^{T}})^{T} $$

Substituting in for $M$:

$$\partial{R} = \partial{\Sigma_{3D}}RSS^{T} + (\partial{\Sigma_{3D}})^{T}RSS^{T}$$ 

$$\partial{S} = R^{T}\partial{\Sigma_{3D}}RS + R^{T}(\partial{\Sigma_{3D}})^{T}RS$$

The expression can be further simplified since $S$, $\Sigma_{3D}$, and $\partial{\Sigma_{3D}}$ are all symmetric:
$$\partial{R} = 2\partial{\Sigma_{3D}}RSS$$

$$\partial{S} = 2R^{T}\partial{\Sigma_{3D}}RS$$


#### 2D Covariance Matrix/Conic
Using the First Quadratic Form from _An extended collection of matrix derivative results_ in section 2.3.2: 

$$C=B^{T}AB$$

$$\partial{A} = B\partial{C}B^{T}$$
$$\partial{B} = AB(\partial{C})^{T} + A^{T}B\partial{C}$$

Subsitituting in the 2D covariance matrix calculation: 

$$\Sigma_{2D} = JW\Sigma{3D}(JW)^T$$

$$A = \Sigma_{3D} $$

$$B = (JW)^T $$


$$\partial{\Sigma_{3D}} = (JW)^{T}\partial{\Sigma_{2D}}JW$$


$$\partial{(JW)^{T}} = \Sigma_{3D}(JW)^{T}(\partial{\Sigma_{2D}})^{T} + \Sigma_{3D}^{T}(JW)^{T} \partial{\Sigma_{2D}}$$ 

With symmetric $\Sigma_{3D}$ and $\partial{\Sigma_{2D}}$:

$$ \partial{(JW)^{T}} = 2\Sigma_{3D}(JW)^{T} \partial{\Sigma_{2D}}$$

Computing the gradient with respect to $J$ and $W$:

$$(JW)^{T} = W^TJ^T$$

$$C = (JW)^{T}$$

$$A = W^T$$ 

$$B = J^T$$

$$\partial{W^T} = \partial{(JW)^{T}} (J^T)^T = \partial{(JW)^{T}}J$$

$$\partial{J^T} = (W^T)^T\partial{(JW)^{T}} = W\partial{(JW)^{T}} $$


Transposing and substituting back in:

$$ \partial{W} = (2\Sigma_{3D}(JW)^{T} \partial{\Sigma_{2D}}J)^T = 2J^T\partial{\Sigma_{2D}}JW\Sigma_{3D}$$

$$ \partial{J} = (2W\Sigma_{3D}(JW)^{T} \partial{\Sigma_{2D}})^T = 2\partial{\Sigma_{2D}}JW\Sigma_{3D}W^T$$ 

