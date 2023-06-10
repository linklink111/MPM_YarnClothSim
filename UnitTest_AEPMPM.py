import os
import bpy
import sys
import numpy as np
import traceback

import random
import math

import taichi as ti
import time

GetMesh = bpy.data.texts["Get_Mesh"].as_module()

ti.init(arch=ti.gpu)



@ti.data_oriented
class AEPMPM:
    def __init__(self, N_Line: int, n_type2: int, n_type3:int, num_rigid_faces:int, num_rigid_vertices:int, num_rigid_edges:int,pin_num:int,main_num_frames:int,sub_num_frames:int):
        
        self.dim = 3
        self.scale_grid = 20.0
        self.N_Line = N_Line
        self.curvePointNums = ti.field(int, shape =self.N_Line)
        self.pin_num = pin_num
        self.Circle_Center = ti.Vector([5.71376,5.0081,6.9932])
        self.main_num_frames = main_num_frames
        self.sub_num_frames = sub_num_frames
        # self.animation_data = ti.Vector.field(3, dtype = float, shape=4000)
        self.current_frame = ti.field(int, shape =())

        self.n_grid = 256
        self.dx = self.scale_grid / self.n_grid
        self.inv_dx = 1.0 / self.dx
        self.dt = 2.5e-4

        self.p_rho = 100 # 
        self.radius = 0.03 # 0.03m

        # Material Parameters
        self.gamma = 100  # shearing stifness # history : 100
        self.k = 500  # stifness  history: 500
        self.youngs_modulu = 500 # 1000: spring force no effect
        self.poisson_ratio = 0.3
        self.lambda_ = self.youngs_modulu * self.poisson_ratio / \
            ((1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio))
        self.mu = self.youngs_modulu / (2 * (1 + self.poisson_ratio))
        
        # rigid objects data
        self.num_rigid_faces = num_rigid_faces
        self.num_rigid_vertices = num_rigid_vertices
        self.num_rigid_edges = num_rigid_edges
        self.rigid_faces = ti.Vector.field(3,int, shape=num_rigid_faces)
        self.rigid_vertices = ti.Vector.field(3,float, shape=num_rigid_vertices)
        self.rigid_edges = ti.Vector.field(2,int, shape=num_rigid_edges)
        self.rigid_normals = ti.Vector.field(3,float, shape=num_rigid_faces)
        self.rigid_centers = ti.Vector.field(3,float, shape=num_rigid_faces)
        self.pin = ti.field(int, shape=pin_num)

        # type2 particle count per line
        self.n_type2 = n_type2
        self.n_type3 = n_type3

        # type2
        self.x2 = ti.Vector.field(
            3, dtype=float, shape=self.n_type2)  # position
        self.v2 = ti.Vector.field(
            3, dtype=float, shape=self.n_type2)  # velocity
        # affine velocity field
        self.C2 = ti.Matrix.field(3, 3, dtype=float, shape=self.n_type2)
        # total volume divided by number of type 2 particles
        self.volume2 = ti.field(ti.f32, shape=())
        self.sl = ti.field(ti.f32, shape=())

        # type3
        self.x3 = ti.Vector.field(
            3, dtype=float, shape=self.n_type3)  # position
        self.v3 = ti.Vector.field(
            3, dtype=float, shape=self.n_type3)  # velocity
        # affine velocity field
        self.C3 = ti.Matrix.field(3, 3, dtype=float, shape=self.n_type3)
        self.F3 = ti.Matrix.field(
            3, 3, dtype=float, shape=self.n_type3)  # deformation gradient
        self.D3_inv = ti.Matrix.field(3, 3, dtype=float, shape=self.n_type3)
        self.d3 = ti.Matrix.field(3, 3, dtype=float, shape=self.n_type3)
        self.volume3 = ti.field(ti.f32, shape=())
        
        
        self.pressure = ti.field(ti.f32, shape=self.n_type3)
        self.listener = ti.Vector([3.0, 3.0, 3.0])
        self.D = ti.Matrix.field(3, 3, dtype=float, shape=self.n_type3)
        self.t = ti.field(ti.f32, shape=self.n_type3)
        self.sample_rate = 44100

        self.grid_v = ti.Vector.field(3, dtype=float, shape=(
            self.n_grid, self.n_grid, self.n_grid))
        self.grid_m = ti.field(dtype=float, shape=(
            self.n_grid, self.n_grid, self.n_grid))
        self.grid_f = ti.Vector.field(3, dtype=float, shape=(
            self.n_grid, self.n_grid, self.n_grid))

        self.n_segment = self.n_type3

        self.cf = 0.05
        self.Phi_F_ = 10  # friction angle in radians
        self.degrees = 90
        self.Phi_F = math.radians(self.degrees)
        self.alpha = (2 / 3) ** 0.5 * (2 * math.sin(self.Phi_F)
                                       ) / (3 - math.sin(self.Phi_F))
        self.beta = math.tan(self.Phi_F)

        self.bound = 3

        self.ROT90_X = ti.Matrix([[1, 0, 0],
                                  [0, 0, -1],
                                  [0, 1, 0]])

        self.ROT90_Y = ti.Matrix([[0, 0, 1],
                                  [0, 1, 0],
                                  [-1, 0, 0]])

        self.ROT90_Z = ti.Matrix([[0, -1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]])

    @ti.func
    def QR3(self,Mat):  # 3x3 mat, Gram–Schmidt Orthogonalization
        c0 = ti.Vector([Mat[0, 0], Mat[1, 0], Mat[2, 0]])
        c1 = ti.Vector([Mat[0, 1], Mat[1, 1], Mat[2, 1]])
        c2 = ti.Vector([Mat[0, 2], Mat[1, 2], Mat[2, 2]])
        r11 = c0.norm(1e-6)
        q0 = c0/r11
        r12 = c1.dot(q0)
        r13 = c2.dot(q0)
        q1 = c1 - r12 * q0
        q2 = c2 - r13 * q0
        r22 = q1.norm(1e-6)
        r23 = q2.dot(q1)
        q1 /= r22
        q2 = q2 - r23*q1
        r33 = q2.norm(1e-6)
        q2 /= r33
        Q = ti.Matrix.cols([q0, q1, q2])
        R = ti.Matrix([[r11, r12, r13], [0, r22, r23], [0, 0, r33]])
        return Q, R

    @ti.kernel
    def Particle_To_Grid(self):
        for p in self.x2:
            base = (self.x2[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x2[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2,
                 0.5 * (fx - 0.5) ** 2]
            affine = self.C2[p]
            mass = self.volume2 * self.p_rho
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                weight = w[i][0] * w[j][1] * w[k][2]
                self.grid_m[base + offset] += weight * mass
                dpos = (offset.cast(float) - fx) * self.dx
                self.grid_v[base + offset] += weight * mass * \
                    (self.v2[p] + affine@dpos)

        for p in self.x3:
            base = (self.x3[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x3[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)
                 ** 2, 0.5 * (fx - 0.5) ** 2]
            affine = self.C3[p]
            mass = self.volume3 * self.p_rho
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                weight = w[i][0] * w[j][1] * w[k][2]
                self.grid_m[base + offset] += weight * mass
                dpos = (offset.cast(float) - fx) * self.dx
                self.grid_v[base + offset] += weight * \
                    mass * (self.v3[p] + affine@dpos)

    # get type2 from type3
    @ti.func
    def GetType2FromType3(self, index):
        left_index = 0
        cnt = 0
        for i in range(self.N_Line):
            left_index += self.curvePointNums[i]
            if left_index > index:
                left_index -= self.curvePointNums[i]
                break
            cnt += 1
        index += cnt
        return index, index+1


    @ti.kernel
    def Grid_Force(self):
        for p in self.x3:
            # get index of type 2 and type 2 particle in mesh(curve)
            l, n = self.GetType2FromType3(p)

            # compute grid position
            base = (self.x3[p] * self.inv_dx - 0.5).cast(int)
            # compute particle position offset from grid
            fx = self.x3[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0)      # compute weight for the particle
                 ** 2, 0.5 * (fx - 0.5) ** 2]

            dw_dx_d = ti.Matrix.rows(
                [fx-1.5, 2*(1.0-fx), fx-0.5,
                 fx-1.5, 2*(1.0-fx), fx-0.5,
                 fx-1.5, 2*(1.0-fx), fx-0.5]) * self.inv_dx

            base_l = (self.x2[l] * self.inv_dx - 0.5).cast(int)
            fx_l = self.x2[l] * self.inv_dx - base_l.cast(float)
            w_l = [0.5 * (1.5 - fx_l) ** 2, 0.75 - (fx_l - 1.0)
                   ** 2, 0.5 * (fx_l - 0.5) ** 2]

            base_n = (self.x2[n] * self.inv_dx - 0.5).cast(int)
            fx_n = self.x2[n] * self.inv_dx - base_n.cast(float)
            w_n = [0.5 * (1.5 - fx_n) ** 2, 0.75 - (fx_n - 1.0)
                   ** 2, 0.5 * (fx_n - 0.5) ** 2]

            Q, R = self.QR3(self.F3[p])

            r11 = R[0, 0]
            r12 = R[0, 1]
            r13 = R[0, 2]
            r22 = R[1, 1]
            r23 = R[1, 2]
            r32 = R[2, 1]
            r33 = R[2, 2]
            # f = (k/2)*(r11-1)**2
            # g = (gamma/2)*(r12**2+r13**2)

            f_derivative = self.k*(r11-1)
            g_derivative = ti.Vector([self.gamma*r12, self.gamma*r13])

            r = ti.Vector([r12, r13])
            rr = r.dot(r)

            # to construct the upper triangular part of A
            R3 = ti.Matrix([[r22, r23], [0, r33]])
            Q1, S, VT = ti.svd(R3, ti.f32)
            eps1 = ti.log(S[0, 0])
            eps2 = ti.log(S[1, 1])
            dh_dR3 = ti.Matrix([[2*self.mu*eps1 + self.lambda_*eps1, 0],
                                [0, 2*self.mu*eps2 + self.lambda_*eps2]])

            dgr_RT = g_derivative.transpose() @ R3

            # construct the upper triangular part of A

            # below is Curve 3D's stress tensor ===================================================================
            A = ti.Matrix([[f_derivative * r11 + g_derivative.dot(r), dgr_RT[0,0], dgr_RT[0,1]],
                           [dgr_RT[0,0], r22*dh_dR3[0, 0], r23*dh_dR3[0, 1]],
                           [dgr_RT[0,1], r32*dh_dR3[1, 0], r33*dh_dR3[1, 1]]])

            # above is Curve 2D's stress tensor ===================================================================

            dphi_dF = Q @ A @ R.inverse().transpose()

            dp_c1 = ti.Vector(
                [self.d3[p][0, 1], self.d3[p][1, 1], self.d3[p][2, 1]])
            dp_c2 = ti.Vector(
                [self.d3[p][0, 2], self.d3[p][1, 2], self.d3[p][2, 2]])
            dphi_dF_c1 = ti.Vector(
                [dphi_dF[0, 1], dphi_dF[1, 1], dphi_dF[2, 1]])
            dphi_dF_c2 = ti.Vector(
                [dphi_dF[0, 2], dphi_dF[1, 2], dphi_dF[2, 2]])
            Dp_inv_c0 = ti.Vector(
                [self.D3_inv[p][0, 0], self.D3_inv[p][1, 0], self.D3_inv[p][2, 0]])

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])

                # technical document .(15) part 1
                weight_l = w_l[i][0] * w_l[j][1] * w_l[k][2]
                weight_n = w_n[i][0] * w_n[j][1] * w_l[k][2]
                f_2 = dphi_dF @ Dp_inv_c0
                self.grid_f[base_l + offset] += self.volume2 * weight_l * f_2
                self.grid_f[base_n + offset] += -self.volume2 * weight_n * f_2

                # dphi w / x
                # dw_dx = ti.Vector(
                #     [dw_dx_d[i, 0] * w[j][1], w[i][0] * dw_dx_d[j, 1]])
                dw_dx = ti.Vector(
                    [dw_dx_d[i, 0] * w[j][1] * w[k][2], w[i][0] * dw_dx_d[j, 1] * w[k][2], w[i][0] * w[j][1] * dw_dx_d[k, 2]])
                # technical document .(15) part 2
                self.grid_f[base + offset] += -self.volume3 * \
                     (dphi_dF_c1*dw_dx.dot(dp_c1)+dphi_dF_c2*dw_dx.dot(dp_c2))

        # spring force, bending parameter
        for nl in range(self.N_Line):
            num_points = self.curvePointNums[nl]
            
            start_index = 0
            for i in range(nl):
                start_index+= self.curvePointNums[i]
                
            for p in range(num_points - 2):
                v0 = p + start_index
                v1 = v0 + 2

                base_0 = (self.x2[v0] * self.inv_dx - 0.5).cast(int)
                fx_0 = self.x2[v0] * self.inv_dx - base_0.cast(float)
                w_0 = [0.5 * (1.5 - fx_0) ** 2, 0.75 - (fx_0 - 1.0) ** 2, 0.5 * (fx_0 - 0.5) ** 2]

                base_1 = (self.x2[v1] * self.inv_dx - 0.5).cast(int)
                fx_1 = self.x2[v1] * self.inv_dx - base_1.cast(float)
                w_1 = [0.5 * (1.5 - fx_1) ** 2, 0.75 - (fx_1 - 1.0) ** 2, 0.5 * (fx_1 - 0.5) ** 2]

                dir_x = self.x2[v1] - self.x2[v0]
                dist = dir_x.norm(1e-9)
                dir_x /= dist
                fn = dist - self.sl # it may cause our previous errors
                f = -1000 * fn * dir_x

                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k])

                    weight_0 = w_0[i][0] * w_0[j][1] * w_0[k][2]
                    weight_1 = w_1[i][0] * w_1[j][1] * w_0[k][2]

                    self.grid_f[base_0 + offset] -= weight_0 * f
                    self.grid_f[base_1 + offset] += weight_1 * f

    @ti.func
    def is_point_inside_triangle(self, point, v0, v1, v2):
        # Calculate the barycentric coordinates of the point with respect to the triangle
        u = v1 - v0
        v = v2 - v0
        w = point - v0
        v_cross_w = v.cross(w)
        u_cross_w = u.cross(w)
        res = False
        if v_cross_w.dot(u_cross_w) < 0:
            res = False
        u_dot_u = u.dot(u)
        u_dot_v = u.dot(v)
        v_dot_v = v.dot(v)
        u_dot_w = u.dot(w)
        v_dot_w = v.dot(w)
        denom = u_dot_u * v_dot_v - u_dot_v * u_dot_v
        s = (u_dot_v * v_dot_w - v_dot_v * u_dot_w) / denom
        t = (u_dot_v * u_dot_w - u_dot_u * v_dot_w) / denom
        
        if s < 0 or t < 0 or s + t > 1:
            res = False
        else:
            res = True
        return res

    
    @ti.kernel
    def Grid_Collision(self):
        for i, j, k in self.grid_m:
            if self.grid_m[i, j, k] > 0.001:
                dv = self.grid_f[i, j, k] * self.dt
                if dv[0] > self.dx*10 or dv[0] < -self.dx*10:
                    dv[0] = 0
                    
                if dv[1] > self.dx*10 or dv[1] < -self.dx*10:
                    dv[1] = 0
                    
                if dv[2] > self.dx*10 or dv[2] < -self.dx*10:
                    dv[2] = 0
                    
                    
                if not (dv[0] < 0 or 0 <dv[0] or dv[0] == 0):
                    dv[0] = 0
                    
                if not (dv[1] < 0 or 0 < dv[1] or dv[1] == 0):
                    dv[1] = 0
                    
                if not (dv[2] < 0 or 0 < dv[2] or dv[2] == 0):
                    dv[2] = 0
                    
                
                
                self.grid_v[i, j, k] += dv
                self.grid_v[i, j, k] /= self.grid_m[i, j, k]
                self.grid_v[i, j, k].z -= 9.8*self.dt
                
                
                # rigid body collision
                mesh_collision = False
                if mesh_collision:
                    grid_pos = ti.Vector([i * self.dx, j * self.dx, k * self.dx])
                    # Define a sphere around the grid node
                    # sphere_radius = self.dx*5 # Adjust the sphere radius as needed
                    sphere_radius = (self.rigid_vertices[0] - self.rigid_vertices[1]).norm()
                    sphere_center = grid_pos
                    num_intersections = 0
                    for f in range(len(self.rigid_faces)):
                        # loop the triangle meshes
                        face = self.rigid_faces[f]
                        v0 = self.rigid_vertices[face[0]]
                        v1 = self.rigid_vertices[face[1]]
                        v2 = self.rigid_vertices[face[2]]
                        face_center = self.rigid_centers[f]
                        # Check if the sphere intersects with the triangle
                        collide = False
                        # Calculate the normal vector of the triangle
                        # normal = (v1 - v0).cross(v2 - v0).normalized()
                        normal = self.rigid_normals[f]
                        # Calculate the distance between the sphere center and the triangle plane
                        dist = (sphere_center - face_center).norm()
                        # Check if the sphere overlaps with the triangle plane
                        if dist < sphere_radius:
                            print("collide")
                            # Project the sphere center onto the triangle plane
                            proj_center = sphere_center - dist*normal
                            # Check if the projected point is inside the triangle
                            # if self.is_point_inside_triangle(proj_center, v0, v1, v2):
                            if True:
                                print(dist)
                                collide = True
                                collision_normal = (sphere_center - proj_center).normalized()
                                # Apply the collision impulse to the sphere and the triangle
                                dv = collision_normal * min(0, self.grid_v[i,j,k].dot(collision_normal))
                                # print(dv)
                                self.grid_v[i, j, k] -= dv
                                self.grid_v[i, j, k] *= 0.9 #friction

                
                # circle collision
                sphere_collision = True
                ground_collision = True
                if sphere_collision:
                    Circle_Center = ti.Vector([5.7071,5.1988,6.5602])   #self.animation_data[self.current_frame] 
                    Circle_Radius = 0.452
                    dist = ti.Vector([i * self.dx, j * self.dx, k * self.dx]) - Circle_Center
                    if dist.x**2 + dist.y**2 + dist.z**2 < Circle_Radius * Circle_Radius:
                        dist = dist.normalized()
                        self.grid_v[i, j, k] -= dist * min(0, self.grid_v[i, j, k].dot(dist))
                        self.grid_v[i, j, k] *= 0.9  # friction
                        
                #ground collision
                if ground_collision:
                    ground_height = 5.0
                    dist_to_ground = k*self.dx - ground_height
                    if dist_to_ground < self.dx:
                        direction = ti.Vector([0.0,0.0,1.0])
                        self.grid_v[i, j, k] -= direction * min(0, self.grid_v[i, j, k].dot(direction))
                        self.grid_v[i, j, k] *= 0.9  # friction
                
                

                if i < self.bound and self.grid_v[i, j, k].x < 0:
                    self.grid_v[i, j, k].x = 0
                if i > self.n_grid - self.bound and self.grid_v[i, j, k].x > 0:
                    self.grid_v[i, j, k].x = 0
                if j < self.bound and self.grid_v[i, j, k].y < 0:
                    self.grid_v[i, j, k].y = 0
                if j > self.n_grid - self.bound and self.grid_v[i, j, k].y > 0:
                    self.grid_v[i, j, k].y = 0
                if k < self.bound and self.grid_v[i, j, k].z < 0:
                    self.grid_v[i, j, k].z = 0
                if k > self.n_grid - self.bound and self.grid_v[i, j, k].z > 0:
                    self.grid_v[i, j, k].z = 0

    @ti.kernel
    def Grid_To_Particle(self):
        for p in self.x2:
            isPin = False
            for pi in range(self.pin_num):
                if self.pin[pi] == p:
                    isPin = True
                    break
            if isPin:
                self.x2[p].x += 0.0001
                continue
            base = (self.x2[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x2[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0)
                 ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(float, 3)
            new_C = ti.Matrix.zero(float, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                dpos = ti.Vector([i, j, k]).cast(float) - fx
                g_v = self.grid_v[base + ti.Vector([i, j, k])]
                weight = w[i][0] * w[j][1] * w[k][2]
                delta_new_v = weight * g_v
                if not (delta_new_v[0] < 0 or 0 < delta_new_v[0] or delta_new_v[0] == 0):
                    delta_new_v[0] = 0.0
                if not (delta_new_v[1] < 0 or 0 < delta_new_v[1] or delta_new_v[1] == 0):
                    delta_new_v[1] = 0.0
                if not (delta_new_v[2] < 0 or 0 < delta_new_v[2] or delta_new_v[2] == 0):
                    delta_new_v[2] = 0.0
                
                #                if delta_new_v[0] > 5000 or delta_new_v[0] < -5000:
                #                    delta_new_v[0] = 0.0
                #                if delta_new_v[1] > 5000 or delta_new_v[1] < -5000:
                #                    delta_new_v[1] = 0.0
                #                if delta_new_v[2] > 5000 or delta_new_v[2] < -5000:
                #                    delta_new_v[2] = 0.0
                new_v += delta_new_v
                new_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx
                
            self.v2[p] = new_v
            delta_x = self.dt * self.v2[p]
            if not (delta_x[0] < 0 or 0 < delta_x[0] or delta_x[0] == 0):
                    delta_x[0] = 0
            if not (delta_x[1] < 0 or 0 < delta_x[1] or delta_x[1] == 0):
                    delta_x[1] = 0
            if not (delta_x[2] < 0 or 0 < delta_x[2] or delta_x[2] == 0):
                    delta_x[2] = 0
            if delta_x[0] > 0.3 * self.dx or delta_x[0] < -0.3 *self.dx:
                delta_x[0] = 0.0
            if delta_x[1] > 0.3 *self.dx or delta_x[1] < -0.3 *self.dx:
                delta_x[1] = 0.0
            if delta_x[2] > 0.3 *self.dx or delta_x[2] < -0.3 *self.dx:
                delta_x[2] = 0.0
            self.x2[p] += delta_x
            self.C2[p] = new_C

        for p in self.x3:
            base = (self.x3[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x3[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0)
                 ** 2, 0.5 * (fx - 0.5) ** 2]
            new_C = ti.Matrix.zero(float, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                dpos = ti.Vector([i, j, k]).cast(float) - fx
                g_v = self.grid_v[base + ti.Vector([i, j, k])]
                
                weight = w[i][0] * w[j][1] * w[k][2]
                new_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx
            self.C3[p] = new_C

    @ti.kernel
    def Update_Particle_State(self):
        for nl in range(self.N_Line):
            num_points = self.curvePointNums[nl]
            
            start_index = 0
            for i in range(nl):
                start_index+= self.curvePointNums[i]
            for p in range(num_points - 1):
                v0 = p + start_index
                v1 = v0 + 1
                dz = (self.x2[v1] - self.x2[v0]).norm()
                self.D[p + start_index] = 2 * np.pi * self.radius**2 * dz * \
                ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        c = 340
        rho = 1.29 # This is rho of air 
        dt = 1e-4
        sample_rate = 44100

        
        for p in self.x3:
            l, n = self.GetType2FromType3(p)  # get type ii particle indices

            # update position and velocity of type iii particle
            v3_last = self.v3[p]
            
            self.v3[p] = 0.5 * (self.v2[l] + self.v2[n])
            self.x3[p] = 0.5 * (self.x2[l] + self.x2[n])
            

            # update deformation record for type iii particle
            dp0 = self.x2[n] - self.x2[l]  # direction D1
            if not (dp0[0] < 0 or 0 < dp0[0] or dp0[0] == 0):
                    dp0[0] = 0.0
            if not (dp0[1] < 0 or 0 < dp0[1] or dp0[1] == 0):
                    dp0[1] = 0.0
            if not (dp0[2] < 0 or 0 < dp0[2] or dp0[2] == 0):
                    dp0[2] = 0.0
            dp1 = ti.Vector([self.d3[p][0, 1], self.d3[p][1, 1], self.d3[p][2,1]])  # direction D2
            ddp1 = dt * self.C3[p] @ dp1
            if not (ddp1[0] < 0 or 0 < ddp1[0] or ddp1[0] == 0):
                    ddp1[0] = 0.0
            if not (ddp1[1] < 0 or 0 < ddp1[1] or ddp1[1] == 0):
                    ddp1[1] = 1.0
            if not (ddp1[2] < 0 or 0 < ddp1[2] or ddp1[2] == 0):
                    ddp1[2] = 0.0
            dp1 += ddp1
            dp2 = ti.Vector([self.d3[p][0, 2], self.d3[p][1, 2], self.d3[p][2,2]])  # direction D3
            ddp2 = dt * self.C3[p] @ dp2
            if not (ddp2[0] < 0 or 0 < ddp2[0] or ddp2[0] == 0):
                    ddp2[0] = 0.0
            if not (ddp2[1] < 0 or 0 < ddp2[1] or ddp2[1] == 0):
                    ddp2[1] = 0.0
            if not (ddp2[2] < 0 or 0 < ddp2[2] or ddp2[2] == 0):
                    ddp2[2] = 1.0
            dp2 += ddp2
            self.d3[p] = ti.Matrix.cols([dp0, dp1, dp2])
            self.F3[p] = self.d3[p] @ self.D3_inv[p]
            # self.d3[p] += self.dt * self.C3[p] @ self.d3[p]

    @ti.kernel
    def return_mapping_3c(self):
        for i in self.x3:
            Q, R = self.QR3(self.d3[i]) # F3_original seems error

            r11 = R[0, 0]
            r12 = R[0, 1]
            r13 = R[0, 2]
            r22 = R[1, 1]
            r23 = R[1, 2]
            r32 = R[2, 1]
            r33 = R[2, 2]
            # f = (k/2)*(r11-1)**2
            # g = (gamma/2)*(r12**2+r13**2)

            f_derivative = self.k*(r11-1)
            g_derivative = ti.Vector([self.gamma*r12, self.gamma*r13])

            r = ti.Vector([r12, r13])

            # to construct the upper triangular part of A
            R3 = ti.Matrix([[1, 0, 0], [0, r22, r23], [0, 0, r33]])
            U, S, VT = ti.svd(R3, ti.f32)
            eps1 = ti.log(S[0, 0])
            eps2 = ti.log(S[1, 1])

            h_p_R3 = ti.Matrix([[2*self.mu*eps1 + self.lambda_*eps1, 0],
                                [0, 2*self.mu*eps2 + self.lambda_*eps2]])

            R3_2x2 = ti.Matrix([[r22, r23], [0, r33]])

            gr_h_p_R3 = g_derivative.transpose() @ R3_2x2.transpose()

            # construct the upper triangular part of A

            # below is Curve 3D's stress tensor ===================================================================
            A = ti.Matrix([[f_derivative * r11 + g_derivative.transpose() @ r, gr_h_p_R3[0,0], gr_h_p_R3[0,1]],
                           [gr_h_p_R3[0,0], r22*h_p_R3[0, 0], r23*h_p_R3[0, 1]],
                           [gr_h_p_R3[0,1], r32*h_p_R3[1, 0], r33*h_p_R3[1, 1]]])
            #            A = ti.Matrix([[f_derivative * r11 + g_derivative.dot(r), gr_h_p_R3[0,0], gr_h_p_R3[0,1]],
            #                           [gr_h_p_R3[0,0], 1, 0],
            #                           [gr_h_p_R3[0,1], 0, 1]])
                           
            q1 = ti.Vector([Q[0,0], Q[1,0], Q[2,0]])
            q2 = ti.Vector([Q[0,1], Q[1,1], Q[2,1]])
            q3 = ti.Vector([Q[0,2], Q[1,2], Q[2,2]])

            s12 = (q1.transpose() @ A @ q2)[0,0]
            s13 = (q1.transpose() @ A @ q3)[0,0]
            s22 = (q2.transpose() @ A @ q2)[0,0]
            s23 = (q2.transpose() @ A @ q3)[0,0]
            s33 = (q3.transpose() @ A @ q3)[0,0]
            
            s12 = Q[0, 1]
            s13 = Q[0, 2]
            s22 = Q[1, 1]
            s23 = Q[1, 2]
            s33 = Q[2, 2]

            # Perform return mapping on R
            if eps1 + eps2 >= 0:
                eps1 = 0
                eps2 = 0
            else:
                J2 = (s22-s33)**2 + 4*s23
                constrait2 = J2 ** 0.5 + self.alpha*(s22 + s33)/2
                if constrait2 <= 0:
                    pass
                else:
                    eta = (eps1-eps2)/2 + self.alpha * \
                        self.lambda_*(eps1+eps2)/(4*self.mu)
                    eps1 -= eta
                    eps2 += eta  # the paper seems error

            exp_epsilon = ti.Matrix([[ti.exp(eps1), 0, 0],
                                    [0, ti.exp(eps2), 0],
                                    [0, 0, 1]])
            R_new = U @ exp_epsilon @ VT

            constrait3 = (s12**2+s13**2)**0.5 + self.beta*(s22+s33)/2
            if constrait3 <= 0:
                pass
            else:
                zeta = - self.beta * (s22 + s33) / 2 * \
                    (s12**2 + s13**2)**0.5
                
                R_new[0, 1] = zeta*r12
                R_new[0, 2] = zeta*r13
                pass

            # update F3
            self.F3[i] = Q @ R_new
            self.d3[i] = self.F3[i] @ self.D3_inv[i].inverse()
            


    @ti.kernel
    def Reset(self):
        self.current_frame += 1
        for i, j, k in self.grid_m:
            self.grid_v[i, j, k] = [0, 0, 0]
            self.grid_m[i, j, k] = 0
            self.grid_f[i, j, k] = [0.0, 0.0, 0.0]

    @ti.kernel
    def initialize(self, curvePointNums:ti.template(),p2: ti.template(), p3: ti.template(),rigid_faces: ti.template(),rigid_vertices: ti.template(),rigid_edges: ti.template(),rigid_normals: ti.template(),rigid_centers: ti.template(),pin:ti.template()):
        self.current_frame = -1
        #        for i in range(4000):
        #            self.animation_data[i] = ti.Vector([animation_data[i][0], animation_data[i][1], animation_data[i][2]])
        for i in range(self.pin_num):
            self.pin[i] = pin[i]
            print(self.pin[i])
        for i in range(self.num_rigid_faces):
            self.rigid_faces[i] = ti.Vector([rigid_faces[i][0], rigid_faces[i][1], rigid_faces[i][2]])
            self.rigid_normals[i] = ti.Vector([rigid_normals[i][0], rigid_normals[i][1], rigid_normals[i][2]])
            self.rigid_centers[i] = ti.Vector([rigid_centers[i][0], rigid_centers[i][1], rigid_centers[i][2]])
        for i in range(self.num_rigid_vertices):
            self.rigid_vertices[i] = ti.Vector([rigid_vertices[i][0], rigid_vertices[i][1], rigid_vertices[i][2]])
        for i in range(self.num_rigid_edges):
            self.rigid_edges[i] = ti.Vector([rigid_edges[i][0], rigid_edges[i][1]])
            
        for i in range(self.N_Line):
            self.curvePointNums[i] = curvePointNums[i]
        for i in range(self.n_type2):
            self.x2[i] = ti.Vector([p2[i][0], p2[i][1], p2[i][2]])
            self.v2[i] = ti.Matrix([0, 0, 0])
            self.C2[i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            
        self.sl = (self.x2[1] - self.x2[0]).norm()

        for i in range(self.n_segment):
            l, n = self.GetType2FromType3(i)
            self.x3[i] = ti.Vector([p3[i][0], p3[i][1], p3[i][2]])
            self.v3[i] = ti.Matrix([0, 0, 0])
            self.F3[i] = ti.Matrix(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            self.C3[i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

            dp0 = self.x2[n] - self.x2[l]
            if not (dp0[0] < 0 or 0 < dp0[0] or dp0[0] == 0):
                    dp0[0] = 0.00
            if not (dp0[1] < 0 or 0 < dp0[1] or dp0[1] == 0):
                    dp0[1] = 0.00
            if not (dp0[2] < 0 or 0 < dp0[2] or dp0[2] == 0):
                    dp0[2] = 0.00

            dp1 = self.ROT90_Z @ dp0
            dp1 /= dp1.norm(1e-6)
            dp2 = dp1.cross(dp0)
            dp2 /= dp2.norm(1e-6)
            self.d3[i] = ti.Matrix.cols([dp0, dp1, dp2])
            self.D3_inv[i] = self.d3[i].inverse()
            
        #        for p in self.x2:
        #            base = (self.x2[p] * self.inv_dx - 0.5).cast(int)
        #            self.grid_m[base] += 1.0
        #        total_volume = 0.0
        #        for i, j, k in self.grid_m:
        #            if self.grid_m[i,j,k] >= 1:
        #                total_volume += 1.0
                
        # total_volume *= (self.dx ** 3)
        # self.volume2 = 3.1415926 * self.radius**2 * (self.x3[0] - self.x2[0]).norm()
        self.volume2 = self.dx**self.dim
        self.volume3 = self.volume2


# ==================================================

def make_data_for_mpm():
    # Get the active object
    curve_object = bpy.context.active_object

    # Make sure the active object is a curve
    if curve_object.type != 'CURVE':
        raise Exception("Active object is not a curve")
    # Get the curve data
    curve_data = curve_object.data

    # Print the number of splines
    N_line = len(curve_data.splines)
    print("Number of splines:", len(curve_data.splines))

    # Iterate through the splines
    # add all vertice to p2
#    x2 = ti.field(3, dtype=float, shape=n_type2)
    cnt = 0
    curvePointNums = []
    for i, spline in enumerate(curve_data.splines):
        cnt += len(spline.points)
        curvePointNums.append(len(spline.points))
    # Iterate through the splines
    cnt2 = 0
    cnt3 = 0
    pin_num = 0
    
    x2 = []
    x3 = []
    pin = []
    
    for i, spline in enumerate(curve_data.splines):
        # Iterate through the points
        for j, point in enumerate(spline.points):
            # Add control point to x2
            x2.append(point.co[0:3])
            
            if(point.select):
                pin.append(cnt2)
                pin_num += 1
            cnt2 += 1
            
            # Check if it's not the first or last point
            if j < len(spline.points)-1:
                # Add segment center to x3
                segment_center = (point.co + spline.points[j+1].co) / 2
                x3.append(segment_center[0:3])
                cnt3 += 1

    n_type2 = cnt
    n_type3 = cnt-N_line
    x2 = np.array(x2)
    x3 = np.array(x3)
    curvePointNums = np.array(curvePointNums)
    pin = np.array(pin)
    
    return N_line, curvePointNums, n_type2, n_type3, x2, x3, pin_num, pin
    
def update_x2_to_curve(x2):
    curve_object = bpy.context.active_object
    curve_data = curve_object.data
    idx = 0
    for i, spline in enumerate(curve_data.splines):
        for j, point in enumerate(spline.points):
            point.co[0:3] = x2[idx]
            idx += 1

main_num_frames = 400
sub_num_frames = 10

N_Line, curvePointNums, n_type2, n_type3, x2, x3,pin_num, pin = make_data_for_mpm()
rigid_vertices, rigid_faces,  rigid_edges = GetMesh.get_mesh("Sphere")
rigid_normals,rigid_centers = GetMesh.get_normals("Sphere")
num_rigid_faces = len(rigid_faces)
num_rigid_vertices = len(rigid_vertices)
num_rigid_edges = len(rigid_edges)
num_rigid_tris = len(rigid_normals)
# Specify the file path
# file_path = "rigid_body_data/positions.csv"

# Load the CSV file into a NumPy array
# animation_data = np.genfromtxt(file_path, delimiter=",")


#with open(f"rigid_body_data/sphere_vert.txt",'w')  as f:
#        for p in rigid_vertices:
#            f.write(str(p)+'\n')
#with open(f"rigid_body_data/sphere_face.txt",'w')  as f:
#        for p in rigid_faces:
#            f.write(str(p)+'\n')
#with open(f"rigid_body_data/sphere_edge.txt",'w')  as f:
#        for p in rigid_edges:
#            f.write(str(p)+'\n')
ti_curvePointNums = ti.field(int, shape=curvePointNums.shape)
ti_x2 = ti.Vector.field(3,float, shape=n_type2)
ti_x3 = ti.Vector.field(3,float, shape=n_type3)


ti_rigid_faces = ti.Vector.field(3,int, shape=num_rigid_faces)
ti_rigid_vertices = ti.Vector.field(3,float, shape=num_rigid_vertices)
ti_rigid_edges = ti.Vector.field(2,int, shape=num_rigid_edges)
ti_rigid_normals = ti.Vector.field(3,int, shape=num_rigid_faces)
ti_rigid_centers = ti.Vector.field(3,int, shape=num_rigid_faces)
ti_pin = ti.field(int, shape=pin_num)
# ti_animation_data = ti.Vector.field(3, float, shape=4000)

aepMPM = AEPMPM(N_Line, n_type2, n_type3, num_rigid_faces, num_rigid_vertices, num_rigid_edges,pin_num,main_num_frames, sub_num_frames)

ti_curvePointNums.from_numpy(curvePointNums)
ti_x2.from_numpy(x2)
ti_x3.from_numpy(x3)

ti_rigid_faces.from_numpy(rigid_faces)
ti_rigid_vertices.from_numpy(rigid_vertices)
ti_rigid_edges.from_numpy(rigid_edges)
ti_rigid_normals.from_numpy(rigid_normals)
ti_rigid_centers.from_numpy(rigid_centers)
ti_pin.from_numpy(pin)
# ti_animation_data.from_numpy(animation_data)
#ti.init()
#print(ti_x2)

aepMPM.initialize(ti_curvePointNums, ti_x2,ti_x3, ti_rigid_faces,ti_rigid_vertices,ti_rigid_edges,ti_rigid_normals,ti_rigid_centers,ti_pin)
for j in range(main_num_frames):
    print(f"frame{j}")
    for i in range(sub_num_frames):
        aepMPM.Reset()
        aepMPM.Particle_To_Grid()
        aepMPM.Grid_Force()
        aepMPM.Grid_Collision()
        aepMPM.Grid_To_Particle()
        aepMPM.Update_Particle_State()
        # aepMPM.return_mapping_3c()
        

    x2_from_ti = aepMPM.x2.to_numpy()  
    pressure_from_ti = aepMPM.pressure.to_numpy()
    t_from_ti = aepMPM.t.to_numpy()
    # for debug
    v2 = aepMPM.v2.to_numpy()
    
    with open(f"work_dir2/pressures/pressure_frame{j}.txt",'w')  as f:
        for p in pressure_from_ti:
            f.write(str(p)+'\n')

    # for debug
    
    

    # update_x2_to_curve(x2_from_ti)
    