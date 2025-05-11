import discr_tools.geometry as geo
import discr_tools.kernels as knls

import numpy as np

import pyopencl.array as cla


class MatvecBase:
    def __init__(self, discr, queue=None):
        if queue is None:
            self._use_gpu_matvec = False
        else:
            self._queue = queue
            self._use_gpu_matvec = True

        self.discr = discr

    def _gpu_matvec(self):
        pass

    def _cpu_matvec(self):
        pass

    def _matvec(self):
        if self._use_gpu_matvec:
            return self._gpu_matvec()
        return self._cpu_matvec()

    def __call__(self, u):
        return self._matvec()(u)


class PoissonMatvec(MatvecBase):
    def _cpu_matvec(self):
        return poisson_matvec(self.discr)

    def _gpu_matvec(self):
        knl = knls.poisson_kernel(self.discr)

        dim, nelts, npts = self.discr.mapped_elements.shape

        if dim == 2:
            npts_1d = int(np.sqrt(npts))
        elif dim == 3:
            npts_1d = int(np.cbrt(npts))
        else:
            raise ValueError("Only supports dim = 2, 3")

        folded_shape = (nelts, *(npts_1d,)*dim)
        unfolded_shape = (nelts, npts)

        d_d = cla.to_device(self._queue, self.discr.operators.diff_operator)

        inv_jac_t = geo.inverse_jacobian_t(self.discr.mapped_elements,
                                           self.discr.basis_cls)
        det_j = geo.jacobian_determinant(self.discr.mapped_elements,
                                         self.discr.basis_cls)

        inv_j_t_tp = inv_jac_t.reshape(dim, dim, *folded_shape, order="F")
        det_j_tp = det_j.reshape(*folded_shape, order="F")

        wts = self.discr.basis_cls.weights

        if dim == 2:
            spec = "kiexy,kjexy,exy,x,y->ijexy"
        elif dim == 3:
            spec = "kiexyz,kjexyz,exyz,x,y,z->ijexyz"
        else:
            raise ValueError("Only supported for dim = 2, 3")

        metrics = np.einsum(spec, inv_j_t_tp, inv_j_t_tp, det_j_tp, *(wts,)*dim)
        metrics = metrics.reshape(-1, order="F")

        metrics_d = cla.to_device(self._queue, metrics)
        metrics_d = metrics_d.reshape(dim, dim, *folded_shape, order="F")

        lap_u_d = cla.zeros(self._queue, unfolded_shape, dtype=np.float64)
        lap_u_d = lap_u_d.reshape(*folded_shape, order="F")

        def matvec(u):
            u_d = cla.to_device(self._queue, self.discr.gather(u))
            u_d = u_d.reshape(*folded_shape, order="F")

            try:
                knl_exec = self._knl_exec
            except AttributeError:
                self._knl_exec = knl.executor(self._queue)
                knl_exec = self._knl_exec

            _, out = knl_exec(
                self._queue,
                u=u_d,
                d=d_d,
                lap_u=lap_u_d,
                metrics=metrics_d
            )

            lap_u = out[0].reshape(*unfolded_shape, order="F")
            return self.discr.scatter(self.discr.apply_mask(lap_u.get()))

        return matvec


def poisson_matvec(discr):
    dim, nelts, npts = discr.mapped_elements.shape
    npts_1d = discr.order + 1

    inv_jac_t = geo.inverse_jacobian_t(discr.mapped_elements, discr.basis_cls)
    det_j = geo.jacobian_determinant(discr.mapped_elements, discr.basis_cls)

    inv_j_t_tp = inv_jac_t.reshape(dim, dim, nelts, *(npts_1d,)*dim, order="F")
    det_j_tp = det_j.reshape(nelts, *(npts_1d,)*dim, order="F")

    wts = discr.basis_cls.weights

    if dim == 2:
        spec = "kiexy,kjexy,exy,x,y->ijexy"
    elif dim == 3:
        spec = "kiexyz,kjexyz,exyz,x,y,z->ijexyz"
    else:
        raise ValueError("Only supported for dim = 2, 3")

    g = np.einsum(spec, inv_j_t_tp, inv_j_t_tp, det_j_tp, *(wts,)*dim)

    def matvec(u):
        u_g = discr.gather(u).reshape(nelts, *(npts_1d,)*dim, order="F")

        d = discr.operators.diff_operator

        if dim == 2:
            ur = np.einsum("il,elj->eij", d, u_g)
            us = np.einsum("jl,eil->eij", d, u_g)

            ux = g[0,0]*ur + g[0,1]*us
            uy = g[1,0]*ur + g[1,1]*us

            uxx = np.einsum("li,elj->eij", d, ux)
            uyy = np.einsum("lj,eil->eij", d, uy)

            lap_u = (uxx + uyy).reshape(nelts, npts, order="F")

        elif dim == 3:
            ur = np.einsum("il,eljk->eijk", d, u_g)
            us = np.einsum("jl,eilk->eijk", d, u_g)
            ut = np.einsum("kl,eijl->eijk", d, u_g)

            ux = g[0,0]*ur + g[0,1]*us + g[0,2]*ut
            uy = g[1,0]*ur + g[1,1]*us + g[1,2]*ut
            uz = g[2,0]*ur + g[2,1]*us + g[2,2]*ut

            uxx = np.einsum("li,eljk->eijk", d, ux)
            uyy = np.einsum("lj,eilk->eijk", d, uy)
            uzz = np.einsum("lk,eijl->eijk", d, uz)

            lap_u = (uxx + uyy + uzz).reshape(nelts, npts, order="F")

        else:
            raise ValueError("Only supported for dim = 2, 3")

        return discr.scatter(discr.apply_mask(lap_u))

    return matvec
