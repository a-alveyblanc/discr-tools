import loopy as lp
import numpy as np

import pyopencl.array as cl_array


def gradient_3d(queue, operator, vec, metrics):
    """
    Register tiled gradient operator.
    """

    nel, npts = vec.shape
    np1d = int(np.cbrt(npts))

    knl = lp.make_kernel(
        "{ [e, i, j, k, l] : 0 <= i, j, k, l < np1d and 0 <= e < nel }",
        [
            # create temporaries to hold non-parallelized dimension
            "".join(f"<> z{i} = u[e, i, j, {i}]\n" for i in range(np1d)),

            # apply operator to scalars of non-parallelized dimension
            "".join(
                f"ut{i}(e, i, j, k) := d[k,{i}] * z{i}\n"
                for i in range(np1d)
            ),

            # reference partials
            """
            ur(e, i, j, k) := sum([l], d[i,l] * u[e,l,j,k])
            us(e, i, j, k) := sum([l], d[j,l] * u[e,i,l,k])
            """,
            "ut(e, i, j, k) := " + "".join(
                f"ut{i}(e, i, j, k) + " for i in range(np1d-1)
            ) + f"ut{np1d-1}(e, i, j, k)",

            # metric term substitution rules
            """
            drdx(e, i, j, k) := G[0, 0, e, i, j, k]
            dsdx(e, i, j, k) := G[0, 1, e, i, j, k]
            dtdx(e, i, j, k) := G[0, 2, e, i, j, k]

            drdy(e, i, j, k) := G[1, 0, e, i, j, k]
            dsdy(e, i, j, k) := G[1, 1, e, i, j, k]
            dtdy(e, i, j, k) := G[1, 2, e, i, j, k]

            drdz(e, i, j, k) := G[2, 0, e, i, j, k]
            dsdz(e, i, j, k) := G[2, 1, e, i, j, k]
            dtdz(e, i, j, k) := G[2, 2, e, i, j, k]
            """

            # apply chain rule and store
            """
            du[0, e, i, j, k] = ur(e, i, j, k) * drdx(e, i, j, k) + \
                                us(e, i, j, k) * dsdx(e, i, j, k) + \
                                ut(e, i, j, k) * dtdx(e, i, j, k)

            du[1, e, i, j, k] = ur(e, i, j, k) * drdy(e, i, j, k) + \
                                us(e, i, j, k) * dsdy(e, i, j, k) + \
                                ut(e, i, j, k) * dtdy(e, i, j, k)

            du[2, e, i, j, k] = ur(e, i, j, k) * drdz(e, i, j, k) + \
                                us(e, i, j, k) * dsdz(e, i, j, k) + \
                                ut(e, i, j, k) * dtdz(e, i, j, k)
            """
        ],
        [
            lp.GlobalArg("u",
                         dim_tags=("N3, N0, N1, N2"),
                         shape=(nel, *(np1d,)*3),
                         dtype=vec.dtype,
                         is_input=True),
            lp.GlobalArg("d",
                         shape=operator.shape,
                         dtype=operator.dtype,
                         is_input=True),
            lp.GlobalArg("du",
                         dim_tags=("N4, N3, N0, N1, N2"),
                         shape=(3, nel, *(np1d,)*3),
                         dtype=vec.dtype,
                         is_output=True),
            lp.GlobalArg("G",
                         dim_tags=("N5, N4, N3, N0, N1, N2"),
                         shape=(3, 3, nel, *(np1d,)*3),
                         dtype=metrics.dtype,
                         is_input=True)
        ]
    )

    knl = lp.fix_parameters(knl, nel=nel, np1d=np1d)
    knl = lp.tag_inames(knl, {
        "e": "g.0",
        "i": "l.0",
        "j": "l.1"
    })

    knl = lp.add_prefetch(
        knl,
        "d[:,:]",
        temporary_address_space=lp.AddressSpace.LOCAL,
        temporary_name="d_s",
        fetch_outer_inames="e",
        default_tag="l.auto"
    )

    knl = lp.add_prefetch(
        knl,
        "u[e,:,:,k]",
        temporary_address_space=lp.AddressSpace.LOCAL,
        temporary_name="u_s",
        default_tag="l.auto",
    )

    knl = lp.add_prefetch(
        knl,
        "G[:,:,e,:,:,k]",
        temporary_address_space=lp.AddressSpace.LOCAL,
        temporary_name="G_s",
        default_tag="l.auto",
    )

    knl = lp.set_instruction_priority(knl, "id:d_fetch", 5)

    # NOTE: awful device copy tricks because of weird strides
    d_d = cl_array.to_device(queue, operator)
    grad_u = cl_array.zeros(queue, (3, *vec.shape), dtype=vec.dtype)
    metrics = cl_array.to_device(queue, metrics.copy())
    u = cl_array.to_device(queue, vec)

    # F-ordered reshapes on DOF axes
    grad_u = grad_u.reshape(3, -1, *(np1d,)*3, order="F")
    metrics = metrics.reshape(3, 3, -1, *(np1d,)*3, order="F")
    u = u.reshape(-1, *(np1d,)*3, order="F")

    knl_exec = knl.executor(queue)
    _, grad_u = knl_exec(queue, u=u, d=d_d, du=grad_u, G=metrics)
    return grad_u[0].reshape(3, -1, npts, order="F").get()


def divergence_3d(queue, operator, vec, metrics):
    """
    Register tiled divergence operator.
    """

    _, nel, npts = vec.shape
    np1d = int(np.cbrt(npts))

    knl = lp.make_kernel(
        "{ [e, i, j, k, l] : 0 <= i, j, k, l < np1d and 0 <= e < nel }",
        [
            # create temporaries to hold non-parallelized dimension
            "".join(f"<> z{i} = u[2, e, i, j, {i}]\n" for i in range(np1d)),

            # apply operator to scalars of non-parallelized dimension
            "".join(
                f"ut{i}(e, i, j, k) := d[k,{i}] * z{i}\n"
                for i in range(np1d)
            ),

            # reference partials
            """
            ur(e, i, j, k) := sum([l], d[i,l] * u[0,e,l,j,k])
            us(e, i, j, k) := sum([l], d[j,l] * u[1,e,i,l,k])
            """,
            "ut(e, i, j, k) := " + "".join(
                f"ut{i}(e, i, j, k) + " for i in range(np1d-1)
            ) + f"ut{np1d-1}(e, i, j, k)",

            # metric term substitution rules
            """
            drdx(e, i, j, k) := G[0, 0, e, i, j, k]
            dsdx(e, i, j, k) := G[0, 1, e, i, j, k]
            dtdx(e, i, j, k) := G[0, 2, e, i, j, k]

            drdy(e, i, j, k) := G[1, 0, e, i, j, k]
            dsdy(e, i, j, k) := G[1, 1, e, i, j, k]
            dtdy(e, i, j, k) := G[1, 2, e, i, j, k]

            drdz(e, i, j, k) := G[2, 0, e, i, j, k]
            dsdz(e, i, j, k) := G[2, 1, e, i, j, k]
            dtdz(e, i, j, k) := G[2, 2, e, i, j, k]
            """

            # apply chain rule and store
            """
            <> dudx = ur(e, i, j, k) * drdx(e, i, j, k) + \
                      us(e, i, j, k) * dsdx(e, i, j, k) + \
                      ut(e, i, j, k) * dtdx(e, i, j, k)

            <> dudy = ur(e, i, j, k) * drdy(e, i, j, k) + \
                      us(e, i, j, k) * dsdy(e, i, j, k) + \
                      ut(e, i, j, k) * dtdy(e, i, j, k)

            <> dudz = ur(e, i, j, k) * drdz(e, i, j, k) + \
                      us(e, i, j, k) * dsdz(e, i, j, k) + \
                      ut(e, i, j, k) * dtdz(e, i, j, k)

            div_u[e, i, j, k] = dudx + dudy + dudz
            """
        ],
        [
            lp.GlobalArg("u",
                         dim_tags=("N4, N3, N0, N1, N2"),
                         shape=(3, nel, *(np1d,)*3),
                         dtype=vec.dtype,
                         is_input=True),
            lp.GlobalArg("d",
                         shape=operator.shape,
                         dtype=operator.dtype,
                         is_input=True),
            lp.GlobalArg("div_u",
                         dim_tags=("N3, N0, N1, N2"),
                         shape=(nel, *(np1d,)*3),
                         dtype=vec.dtype,
                         is_output=True),
            lp.GlobalArg("G",
                         dim_tags=("N5, N4, N3, N0, N1, N2"),
                         shape=(3, 3, nel, *(np1d,)*3),
                         dtype=metrics.dtype,
                         is_input=True)
        ]
    )

    knl = lp.fix_parameters(knl, nel=nel, np1d=np1d)

    knl = lp.tag_inames(knl, {
        "e": "g.0",
        "i": "l.0",
        "j": "l.1"
    })

    knl = lp.add_prefetch(
        knl,
        "d[:,:]",
        temporary_address_space=lp.AddressSpace.LOCAL,
        temporary_name="d_s",
        fetch_outer_inames="e",
        default_tag="l.auto"
    )

    knl = lp.add_prefetch(
        knl,
        "u[:,e,:,:,k]",
        temporary_address_space=lp.AddressSpace.LOCAL,
        temporary_name="u_s",
        default_tag="l.auto",
    )

    knl = lp.add_prefetch(
        knl,
        "G[:,:,e,:,:,k]",
        temporary_address_space=lp.AddressSpace.LOCAL,
        temporary_name="G_s",
        default_tag="l.auto",
    )

    knl = lp.set_instruction_priority(knl, "id:d_fetch", 5)

    d_d = cl_array.to_device(queue, operator)
    metrics = cl_array.to_device(queue, metrics.copy())
    u = cl_array.to_device(queue, vec)
    div_u = cl_array.zeros(queue, (nel, np1d**3), dtype=vec.dtype)

    # F-ordered reshapes on DOF axes
    u = u.reshape(3, -1, *(np1d,)*3, order="F")
    div_u = div_u.reshape(-1, *(np1d,)*3, order="F")
    metrics = metrics.reshape(3, 3, -1, *(np1d,)*3, order="F")

    knl_exec = knl.executor(queue)
    _, div_u = knl_exec(queue, u=u, d=d_d, div_u=div_u, G=metrics)
    return div_u[0].reshape(-1, npts, order="F").get()


def divgrad_3d(queue, operator, vec, metrics):
    return divergence_3d(queue,
                         operator,
                         gradient_3d(queue, operator, vec, metrics),
                         metrics)


