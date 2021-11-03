# The simplest numerical recipes to solve Schrödinger equation in a periodic potential

```python id=75327c66-fbe7-40b9-b9e9-7baaab72472f
import numpy as np
import matplotlib.pyplot as plt
```

The goal of this project is to implement a program which solve a single quantum particle in a stationary one-dimensional potential.

$$
- \frac{\hbar^2}{2m} \frac{d^2 \psi}{dx^2} (x) + V(x) \psi(x) = E \psi(x) \hspace{1cm} (*)
$$
where the lattice-potential is supposed to be periodic : 

$$
\forall x \in \mathbb{R}, V(x+a) = V(x)
$$
where $a$ is the lattice period.

Understanding this kind of potential is the key of the solid-state physics e.g. the description of electron in the crystals. In the description below, we are going to model Hartree electrons in a periodic potential. 

To simplify the notation and ease the code, you might use Hartree atomic units : 

$$
\hbar =1, m_e = 1
$$
# Bloch theorem

The eigensolutions of $(*)$ under a periodic potential are solution of two operators which commutes : 

* the hermitian Hamiltonian,
* the unitary translation operator.

$$
\hat{T}_a \Psi(x) = \Psi(x+a)
$$
We add a boundary condition called the Born-von Karmen boundary condition, supposing it exists $N \in \mathbb{N}$ such as : 

$$
\Psi(x + Na) = \Psi(x)
$$
**Q.1** Apply N times the translation operator. Using the norm conservation by translation, demonstrate the eigensolutions might be written as : 

$$
\psi_k(x+a) = e^{jka} \psi_k(x)
$$
where k is a wave vector called the crystal momentum. Give it's expression.

**Q.2** Demonstrate the bellowing mathematical relation is equal to the one at **Q.1**.

$$
\psi_k(x) = u_k(x) e^{jkx} \text{ with } u_k(x+a) = u_k(x)
$$
# From continuum to finite-difference

## Translate the differential equation

The first-step is to discrete space with a finite-difference method is such a way : 

$$
x \in \mathbb{R} \to x_l = l \cdot\delta x\text{, }l \in \mathbb{Z}
$$
We remember the definition of the derivation of real functions : 

$$
f'(x) = \lim_{\delta x \to 0} \frac{f(x+\delta x) - f(x)}{\delta x} 
$$
**Q.3** Discrete the equation $(*)$ by doing :

$$
\begin{align}
\psi(x_l) & \to \psi_l \\
V(x_l) & \to V_l
\end{align}
$$
**R.3**

**Q.4** Supposing the problem is a repetition of spatially finite cells, composed with N discrete value.  Represents the previous equation with a matrix form :

$$
\underline{H} \underline{\psi} = E \underline{\psi}
$$
with   $\underline{\psi}$ an (N,)-shaped array representing the wave function and $\underline{H}$ an (N,N)-shaped array representing the Hamiltonian. Don't forget the periodic boundary condition.

What kind of matrix do you obtain ?

**R.4**

**Q.5** What are the computational limitations of this approach ?

**R.5** 

*Note : For a complete discussion about the limits of finite difference method, you can have a look in \[1\].*

## Translate the periodic boundary conditions

A second step is to include the periodic boundary conditions in the discretization. We reduce the k-space to the first Brillouin zone, defined as : 

$$
\forall k \in \left[ - \frac{\pi}{a}, \frac{\pi}{a} \right] \iff FBZ
$$
**Q.6**  Rewrite the Hamiltonian in ($*$) to include these conditions. Demonstrate the equation can be written as : 

$$
\forall k \in FBZ, -\frac{1}{2}u''_k(x) - jk u'_k(x) + \frac{1}{2} k^2 u_k(x) + V(x) u_k(x) = E_k(k) u_k(x) \hspace{2cm}(**)
$$
**R.6** 

**Q.7**  Use the finite-difference approximation and write $\underline{U}_k$ an (N,)-shaped array representing the periodic function $u_k(x)$ and $ \underline{H}_k$ an (N,N)-shaped array representing the Hamiltonian.

*Help : Don't forget to add the periodic conditions when you discretize the space.*

**R.7** 

**Q.8** Implement this eigenproblem where a user can control : 

1. the python function defining the lattice potential $V$in one period;
2. the spatial periodicity of the lattice $a$;
3. the value of k;
4. the spatial resolution of your grid $\delta x$.

The code returns a npz file containing the $k$ value, the eigenenergies $E_k$ , the x-axis, the eigenfunctions $\underline{U}_k$. 

```python id=6e9522bc-39c0-4149-91e8-86c436fb16ce
#R.8
```

**Q.9** Run your code for different values of $k$ in the first Brillouin zone for a square potential.

**R.9** 

**Q.10** Write an analysis script which reads the files you generate. Represents the energy profile as a function of $k$ and illustrate the shape of the Bloch waves at these points.

```python id=81c2e411-48d0-44c2-bcbc-48d2888d8c1a
#R. 10
```

*Help : Don't forget to apply Bloch theorem on $\underline{U}_k$.*

A brute force approach is used here! If your lattice is large and is defined in 2/3 dimensions, the calculation might become computationally complicate. In order to build the dispersion relation of the Bloch electron, you might apply multiple times your code. A more clever approach is to use the periodicity of the system by solving numerically in the Fourier space.  

# Solution in the Fourier space

## Translate the problem in the Fourier Space

**Q.11** $u_k$ and $V$ are periodic with period a. One can expand them as Fourier series : 

$$
\begin{align}
u_k(x) &= \sum_{n=-\infty}^{\infty} \chi_{k,n} e^{j \frac{2 \pi n}{a} x} \\
V(x) &= \sum_{n = -\infty}^\infty V_n e^{j \frac{2 \pi n}{a} x}
\end{align}
$$
Adding Bloch theorem and Fourier series on wave function, demonstrate the problem is reduced to an eigenproblem with the form :

$$
\forall n \in \mathbb{Z}, \hspace{1cm} -\frac{1}{2} \chi_{k, n} \left(\frac{2 \pi n}{a} + k\right)^2 + \sum_{n'} \chi_{k,n'} V_{n-n'} = E_k \chi_{k,n}
$$
**R.11** 

**Q.12** State this problem as an infinite-sized eigenproblem, with the form :

$$
\underline{A}_k \underline{X}_k = E_k \underline{X}_k
$$
where an $(\infty, )$-shaped array representing the Fourier coefficients, labeled as $\underline{X}_k$ and an $(\infty,\infty)$-shaped array representing the Hamiltonian in Fourier space, labeled $ \underline{A}_k$. 

**R.12**

**Q.13** From what you know of Fourier transformation, if your function is slowly changing, you might be able to truncate the infinite, in such a way that : 

$$
\infty \to p
$$
p is the maximum order of the Fourier transform. Moreover, you only need to sample a finite number of k value. In such approximations, rewrite **Q.11** and **Q.12**.

**R.13** 

**Q.14** At which condition this approach is more efficient than the brute force method ?

**R.14**

## Code

**Q.15** Implement the problem of **Q.8** in the Fourier space. The input parameters are :  

1. Fourier truncation order;
2. Cell size;
3. Potential function;
4. The recriprocal space resolution.

The npz file might contain the value of the truncation, the k values and their energies with the Fourier coefficients for each value of k.

```python id=63505eae-07ee-4b22-b409-864b4253f6a4
#R.15
```

**Q.16** Write a script which reconstruct the dispersion relation of energy and the Bloch function from the relation defined in **Q.11**.

```python id=40f8e3d9-f065-4dbb-b409-808372f0745a
#R.16
```

**Q.17** How the truncation order modify the solutions ? Sample different values of $p$.

```python id=87f575b0-8e6b-4b16-8931-f048b1899946
#R.17
```

# Open Question

Modify the input and include a square-potential with two atoms (\~ different potential depth and width) in the lattice. How is the energy relation modify if the two atoms are equivalent ? How is the relation modify if one atom has a deeper potential ? Is there an obvious physical interpretation ?

```python id=4ea1a0a9-c9cd-44ec-8c7e-0e15c61a2e9c
#R.18
```

project based on : 

*\[1\] William H. Press & all, Numerical Recipes, The Art of Scientific Computing, third edition*

*\[2\] Based on M2 SMNO lectures*

*\[3\] Neil W. Ashcroft, N. David Mermin, Solid state physics*

*\[4\] James N. Nagel, Numerical Solutions to Poisson Equations Using Finite-Difference Method, IEEE Antennas Propag. Mag. **56**, 209–224 (2014)*

<details id="com.nextjournal.article">
<summary>This notebook was exported from <a href="https://nextjournal.com/a/NBtUMQBxo1uLYGLPNs227?change-id=CqFc9gAS71QDCQg4VTr2hb">https://nextjournal.com/a/NBtUMQBxo1uLYGLPNs227?change-id=CqFc9gAS71QDCQg4VTr2hb</a></summary>

```edn nextjournal-metadata
{:article
 {:nodes
  {"0dd09bbe-df2e-4369-93c1-55d3e2b6e10c"
   {:environment
    [:environment
     {:article/nextjournal.id
      #uuid "5b45e08b-5b96-413e-84ed-f03b5b65bd66",
      :change/nextjournal.id
      #uuid "5f0c0e79-790f-439a-8b18-fb81409f12c2",
      :node/id "0149f12a-08de-4f3d-9fd3-4b7a665e8624"}],
    :id "0dd09bbe-df2e-4369-93c1-55d3e2b6e10c",
    :kind "runtime",
    :language "python",
    :type :nextjournal},
   "40f8e3d9-f065-4dbb-b409-808372f0745a"
   {:id "40f8e3d9-f065-4dbb-b409-808372f0745a",
    :kind "code",
    :runtime [:runtime "0dd09bbe-df2e-4369-93c1-55d3e2b6e10c"]},
   "4ea1a0a9-c9cd-44ec-8c7e-0e15c61a2e9c"
   {:id "4ea1a0a9-c9cd-44ec-8c7e-0e15c61a2e9c",
    :kind "code",
    :runtime [:runtime "0dd09bbe-df2e-4369-93c1-55d3e2b6e10c"]},
   "63505eae-07ee-4b22-b409-864b4253f6a4"
   {:id "63505eae-07ee-4b22-b409-864b4253f6a4",
    :kind "code",
    :runtime [:runtime "0dd09bbe-df2e-4369-93c1-55d3e2b6e10c"]},
   "6e9522bc-39c0-4149-91e8-86c436fb16ce"
   {:id "6e9522bc-39c0-4149-91e8-86c436fb16ce",
    :kind "code",
    :runtime [:runtime "0dd09bbe-df2e-4369-93c1-55d3e2b6e10c"]},
   "75327c66-fbe7-40b9-b9e9-7baaab72472f"
   {:compute-ref #uuid "323525ab-e871-4dfc-8147-e903a2c0993c",
    :exec-duration 689,
    :id "75327c66-fbe7-40b9-b9e9-7baaab72472f",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "0dd09bbe-df2e-4369-93c1-55d3e2b6e10c"]},
   "81c2e411-48d0-44c2-bcbc-48d2888d8c1a"
   {:id "81c2e411-48d0-44c2-bcbc-48d2888d8c1a",
    :kind "code",
    :runtime [:runtime "0dd09bbe-df2e-4369-93c1-55d3e2b6e10c"]},
   "87f575b0-8e6b-4b16-8931-f048b1899946"
   {:id "87f575b0-8e6b-4b16-8931-f048b1899946",
    :kind "code",
    :runtime [:runtime "0dd09bbe-df2e-4369-93c1-55d3e2b6e10c"]}},
  :nextjournal/id #uuid "02f5542b-d70c-4bc3-9e2a-4d7f7f94a884",
  :article/change
  {:nextjournal/id #uuid "5fd13a3d-b224-45aa-b238-0328eda3525e"}}}

```
</details>
