This code is used to get the control inputs needed to achive desired trajectories. for this instance I am using it to get the motor speeds needed to drive a quad-rotor drone along set velocity trajectories. 

# The Dynamics Model
The state vector and its time derivative are defined as follows:

![\mathbf{x} = \begin{bmatrix}
    x \\
    v_x \\
    y \\
    v_y \\
    z \\
    v_z \\
    \phi \\
    \dot{\phi} \\
    \theta \\
    \dot{\theta} \\
    \psi \\
    \dot{\psi}
\end{bmatrix}, \quad
\dot{\mathbf{x}} = \begin{bmatrix}
    \dot{x} \\
    \dot{v_x} \\
    \dot{y} \\
    \dot{v_y} \\
    \dot{z} \\
    \dot{v_z} \\
    \dot{\phi} \\
    \ddot{\phi} \\
    \dot{\theta} \\
    \ddot{\theta} \\
    \dot{\psi} \\
    \ddot{\psi}
\end{bmatrix}](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bx%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20x%20%5C%5C%20v_x%20%5C%5C%20y%20%5C%5C%20v_y%20%5C%5C%20z%20%5C%5C%20v_z%20%5C%5C%20%5Cphi%20%5C%5C%20%5Cdot%7B%5Cphi%7D%20%5C%5C%20%5Ctheta%20%5C%5C%20%5Cdot%7B%5Ctheta%7D%20%5C%5C%20%5Cpsi%20%5C%5C%20%5Cdot%7B%5Cpsi%7D%20%5Cend%7Bbmatrix%7D%2C%20%5Cquad%20%5Cdot%7B%5Cmathbf%7Bx%7D%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdot%7Bx%7D%20%5C%5C%20%5Cdot%7Bv_x%7D%20%5C%5C%20%5Cdot%7By%7D%20%5C%5C%20%5Cdot%7Bv_y%7D%20%5C%5C%20%5Cdot%7Bz%7D%20%5C%5C%20%5Cdot%7Bv_z%7D%20%5C%5C%20%5Cdot%7B%5Cphi%7D%20%5C%5C%20%5Cddot%7B%5Cphi%7D%20%5C%5C%20%5Cdot%7B%5Ctheta%7D%20%5C%5C%20%5Cddot%7B%5Ctheta%7D%20%5C%5C%20%5Cdot%7B%5Cpsi%7D%20%5C%5C%20%5Cddot%7B%5Cpsi%7D%20%5Cend%7Bbmatrix%7D)

# The code

