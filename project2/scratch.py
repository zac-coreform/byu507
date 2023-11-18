import mesh as m
import integrate as i
import basis as b
import assembly as a

m1 = m.generate_mesh(0, 1, 2)
ien = m1[0]
nodes = m1[1]
n_elem = m.get_num_elem_from_ien(ien)
print("n_elem", n_elem)
for e in range(0, n_elem):
    ed = m.get_element_domain(ien, e, nodes)
    print("ed", ed)


for e in range(0, n_elem):
    ke = m.get_element_ke(e=0, ien=ien, nodes=nodes, constit_coeff=1)
    print("ke", ke)


# print(ke)

