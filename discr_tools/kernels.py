import loopy as lp

import numpy as np


def poisson_kernel(discr):
    dim, nelts, npts = discr.mapped_elements.shape

    if dim == 2:
        npts_1d = int(np.sqrt(npts))
    elif dim == 3:
        npts_1d = int(np.cbrt(npts))
    else:
        raise ValueError("Only supports dim = 2, 3")

    knl = lp.make_kernel(
        "{ [e, i, j, k, l, m] : 0 <= i, j, k, l, m< np and 0 <= e < ne }",
        [
            """
            ur(e,i,j,k) := sum([l], d[i,l] * u[e,l,j,k])
            us(e,i,j,k) := sum([l], d[j,l] * u[e,i,l,k])
            """,
            "".join(
                f"<> z{i} = u[e,i,j,{i}]\n"
                for i in range(npts_1d)
            ),
            "".join(
                f"ut{i}(e,i,j,k) := d[k,{i}]*z{i}\n"
                for i in range(npts_1d)
            ),
            "ut(e,i,j,k) := " + (
                "".join(f"ut{i}(e,i,j,k) + " for i in range(npts_1d-1))
                + f"ut{npts_1d-1}(e,i,j,k)"
            ),
            """
            drdx(e,i,j,k) := metrics[0,0,e,i,j,k]
            dsdx(e,i,j,k) := metrics[0,1,e,i,j,k]
            dtdx(e,i,j,k) := metrics[0,2,e,i,j,k]

            drdy(e,i,j,k) := metrics[1,0,e,i,j,k]
            dsdy(e,i,j,k) := metrics[1,1,e,i,j,k]
            dtdy(e,i,j,k) := metrics[1,2,e,i,j,k]

            drdz(e,i,j,k) := metrics[2,0,e,i,j,k]
            dsdz(e,i,j,k) := metrics[2,1,e,i,j,k]
            dtdz(e,i,j,k) := metrics[2,2,e,i,j,k]
            """,
            """
            dudx(e,i,j,k) := drdx(e,i,j,k) * ur(e,i,j,k) + \
                             dsdx(e,i,j,k) * us(e,i,j,k) + \
                             dtdx(e,i,j,k) * ut(e,i,j,k)

            dudy(e,i,j,k) := drdy(e,i,j,k) * ur(e,i,j,k) + \
                             dsdy(e,i,j,k) * us(e,i,j,k) + \
                             dtdy(e,i,j,k) * ut(e,i,j,k)

            dudz(e,i,j,k) := drdz(e,i,j,k) * ur(e,i,j,k) + \
                             dsdz(e,i,j,k) * us(e,i,j,k) + \
                             dtdz(e,i,j,k) * ut(e,i,j,k)
            """,
            """
            uxx(e,i,j,k) := sum([m], d[m,i] * dudx(e,m,j,k))
            uyy(e,i,j,k) := sum([m], d[m,j] * dudy(e,i,m,k))
            uzz(e,i,j,k) := sum([m], d[m,k] * dudz(e,i,j,m))

            lap_u[e,i,j,k] = uxx(e,i,j,k) + uyy(e,i,j,k) + uzz(e,i,j,k)
            """
        ],
        [
            lp.GlobalArg("u",
                         dim_tags=(f"N{dim}, " +
                                   "".join(f"N{i}, " for i in range(dim-1)) +
                                   f"N{dim-1}"),
                         shape=(nelts, *(npts_1d,)*dim),
                         is_input=True
            ),
            lp.GlobalArg("metrics",
                         # dim_tags=(f"N{dim+2}, N{dim+1}, N{dim}," +
                         #           "".join(f"N{i}, " for i in range(dim-1)) +
                         #           f"N{dim-1}"),
                         order="F",
                         shape=(dim, dim, nelts, *(npts_1d,)*dim),
                         is_input=True
            ),
            lp.GlobalArg("d", shape=(npts_1d, npts_1d), is_input=True),
            lp.GlobalArg("lap_u",
                         dim_tags=(f"N{dim}, " +
                                   "".join(f"N{i}, " for i in range(dim-1)) +
                                   f"N{dim-1}"),
                         shape=(nelts, *(npts_1d,)*dim),
                         is_output=True
            )
        ]
    )

    knl = lp.fix_parameters(knl, np=npts_1d, ne=nelts)

    knl = lp.tag_inames(
        knl,
        {
            "e" : "g.0",
            "i" : "l.0",
            "j" : "l.1"
        }
    )

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

    knl = lp.set_instruction_priority(knl, "id:d_fetch", 5)

    return knl


def generic_tensor_contraction(knl, dim, axis):
    operator_spec = "ij"
    data_spec = f"e{"mno"[:axis]}j{"pqr"[:dim-axis-1]}"
    out_spec = f"e{"mno"[:axis]}i{"pqr"[:dim-axis-1]}"

    spec = operator_spec + "," + data_spec + "->" + out_spec

    knl = lp.make_einsum(
        spec=spec,
        arg_names=("d", "u")
    )

    local_inames = set(data_spec) - {"e", "j"}
    tagged_inames = {iname : f"l.{i}" for i, iname in enumerate(local_inames)}
    tagged_inames |= {"e" : "g.0"}

    knl = lp.tag_inames(knl, tagged_inames)

    knl = lp.add_prefetch(
        knl,
        "d[:,:]",
        temporary_address_space=lp.AddressSpace.LOCAL,
        temporary_name="d_s",
        default_tag="l.auto"
    )

    return knl
