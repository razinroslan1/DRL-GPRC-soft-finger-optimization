import math
import numpy as np

def FEM_sim(mapdl,
            chamber_height,
            slit_thickness,
            slit_length,
            gp = None,
            scaler = None,
            use_gprc = False):
    
    """
    Run the nonlinear finite-element model of the Pneunet soft robotic finger.

    Parameters
    ----------
    mapdl: MAPDL solver
        MAPDL object
    chamber_height : float
        Chamber height in mm.
    slit_thickness : float
        Slit thickness in mm.
    slit_length : float
        Slit length in mm.
    gp: GaussianProcessRegressor
        Gaussian Process model.
    scaler: MinMaxScaler
        Fitted scaler for normalizing GPRC input parameters.
    use_gprc: bool
        Enable or disable GPRC.

    Returns
    -------
    theta : float
        Predicted bending angle in degrees.
    """

    if use_gprc and (gp is None or scaler is None):
        raise ValueError("gp and scaler inputs are necessary when use_gprc=True")
    
    mapdl.clear()
    mapdl.prep7()
    mapdl.etcontrol("SET")


    # Simulation parameters
    pressure = 40000 # Pa
    n_chambers = 4 
    element_size = 0.0034 # m
    density = 1040 # kg/m^3
    yeoh_c1 = 127557.442
    yeoh_c2 = 36713.7833
    yeoh_c3 = -12.161666
    yeoh_d1 = 0 
    yeoh_d2 = 0 
    yeoh_d3 = 0

    length = 45
    width = 15 
    slit_bottom_thickness = 1 
    bottom_thickness = 4
    top_thickness = 2 
    side_thickness = 2 
    end_thickness = 3 
    connecting_chamber_height = 2 
    connecting_chamber_width_top = 3 
    connecting_chamber_width_bottom = 7

    node_tolerance = 1e-4 # Tolerance for selecting finger tip node

    # Calculation and modeling of soft finger geometry
    mm2m = 1e-3 # Converting mm to m

    chamber_length = (length - end_thickness * 2 - slit_length * (n_chambers - 1) + slit_thickness * 2) / n_chambers

    height = bottom_thickness + chamber_height + top_thickness

    # Inner chamber initial block
    mapdl.block(x1 = side_thickness * mm2m,
                x2 = (width - side_thickness) * mm2m,
                y1 = end_thickness * mm2m,
                y2 = (length - end_thickness) * mm2m,
                z1 = (height - top_thickness - chamber_height) * mm2m,
                z2 = (height - top_thickness) * mm2m)

    for i in range(n_chambers-1): # cutting inner chamber into correct shape
        ic0 = mapdl.k(x = (width / 2 + connecting_chamber_width_top / 2) * mm2m,
                      y = (i * (chamber_length + slit_length) + end_thickness + chamber_length - 2 * slit_thickness) * mm2m,
                      z = (bottom_thickness + connecting_chamber_height) * mm2m)
        
        ic1 = mapdl.k(x = (width / 2 + connecting_chamber_width_bottom / 2) * mm2m,
                      y = (i * (chamber_length + slit_length) + end_thickness + chamber_length - 2 * slit_thickness) * mm2m,
                      z = bottom_thickness * mm2m)
        
        ic2 = mapdl.k(x = (width - side_thickness) * mm2m,  
                      y = (i * (chamber_length + slit_length) + end_thickness + chamber_length - 2 * slit_thickness) * mm2m,
                      z = bottom_thickness * mm2m)
        
        ic3 = mapdl.k(x = (width-side_thickness) * mm2m,
                      y = (i * (chamber_length + slit_length) + end_thickness + chamber_length - 2 * slit_thickness) * mm2m,
                      z = (bottom_thickness + chamber_height) * mm2m)
        
        ic4 = mapdl.k(x = side_thickness * mm2m,
                      y = (i * (chamber_length  +slit_length) + end_thickness + chamber_length - 2 * slit_thickness) * mm2m,
                      z = (bottom_thickness + chamber_height) * mm2m)
        
        ic5 = mapdl.k(x = side_thickness * mm2m,
                      y = (i * (chamber_length + slit_length) + end_thickness + chamber_length - 2 * slit_thickness) * mm2m,
                      z = bottom_thickness * mm2m)

        ic6 = mapdl.k(x = (width / 2 - connecting_chamber_width_bottom / 2) * mm2m,
                      y = (i * (chamber_length + slit_length) + end_thickness + chamber_length - 2 * slit_thickness) * mm2m,
                      z = bottom_thickness * mm2m)
        
        ic7 = mapdl.k(x = (width / 2 - connecting_chamber_width_top / 2) * mm2m,
                      y = (i * (chamber_length + slit_length) + end_thickness + chamber_length - 2 * slit_thickness) * mm2m,
                      z = (bottom_thickness + connecting_chamber_height) * mm2m)
        
        inner_cut_area = mapdl.a(ic0, ic1, ic2, ic3, ic4, ic5, ic6, ic7)
        mapdl.vext(inner_cut_area,
                   dy = (2 * slit_thickness + slit_length) * mm2m)

        if i == 0: # Check for first iteration as inner chamber block ID changes and location in vnum changes 
            inner_chamber_block = mapdl.geometry.vnum[0]
            inner_cut_block = mapdl.geometry.vnum[1]
        else:
            inner_chamber_block = mapdl.geometry.vnum[1]
            inner_cut_block = mapdl.geometry.vnum[0]

        mapdl.vsbv(nv1 = inner_chamber_block,
                   nv2 = inner_cut_block,
                   keep1 = "DELETE",
                   keep2 = "DELETE")
        
        inner_chamber_block = mapdl.geometry.vnum[0]

    mb0 = mapdl.k(x = 0, y= 0 , z = 0)
    mb1 = mapdl.k(x = 0, y = 0, z = height * mm2m)
    mb2 = mapdl.k(x = 0, y = length * mm2m, z = height * mm2m)
    mb3 = mapdl.k(x = 0, y = length * mm2m, z = 0)
    main_body = mapdl.a(mb0, mb1, mb2, mb3) 

    for i in range(n_chambers - 1): # Cutting out slits from the part
        s0 = mapdl.k(x = 0,
                     y = (i * (chamber_length + slit_length) + end_thickness + chamber_length - slit_thickness) * mm2m,
                     z = (bottom_thickness + connecting_chamber_height + slit_bottom_thickness) * mm2m)
        
        s1 = mapdl.k(x = 0,
                     y = (i * (chamber_length + slit_length) + end_thickness + chamber_length - slit_thickness) * mm2m,
                     z = height * mm2m)
        
        s2 = mapdl.k(x = 0,
                     y=(i * (chamber_length + slit_length) + end_thickness + chamber_length - slit_thickness + slit_length) * mm2m,
                     z = height * mm2m)

        s3 = mapdl.k(x = 0,
                     y = (i * (chamber_length + slit_length) + end_thickness + chamber_length - slit_thickness + slit_length) * mm2m,
                     z = (bottom_thickness + connecting_chamber_height + slit_bottom_thickness) * mm2m)
        
        slit = mapdl.a(s0, s1, s2, s3)
        main_body = mapdl.asba(main_body, slit)

    mapdl.vext(main_body, dx = width * mm2m) 
    main_body = mapdl.geometry.vnum[0] 
    inner_chamber_block = mapdl.geometry.vnum[1]
    mapdl.vsbv(nv1 = main_body, nv2 = inner_chamber_block, keep1 = "DELETE", keep2 = "DELETE")
    pneunet = mapdl.geometry.vnum[0]

    mapdl.units("SI")

    # Meshing
    mapdl.mshkey(0)
    mapdl.mshape(1,"3D")
    mapdl.esize(element_size)
    mapdl.et(1, "SOLID187")
    mapdl.mp("MU",1,0) 
    mapdl.mat(1)
    mapdl.r(3)
    mapdl.real(3)
    mapdl.et(2, "TARGE170")
    mapdl.et(3, "CONTA174")
    mapdl.keyopt(3, 8, 2)     # Automatic asymmetric contact
    mapdl.keyopt(3, 10, 0)    # Adjust contact stiffness each Newton-Raphson iteration 
    mapdl.keyopt(3, 12, 0)    # Standard contact behavior
    mapdl.keyopt(3, 2, 0)     # Augmented Lagrange
    mapdl.keyopt(3, 4, 0)     # On Gauss point 
    mapdl.keyopt(3, 9, 0)     # Use real geometry with stepped effects
    mapdl.keyopt(3, 7, 0)     # No Prediction
    mapdl.rmodif(3, 3, 1)   
    mapdl.rmodif(3, 5, 0)     
    mapdl.rmodif(3, 6, 0)     
    mapdl.rmodif(3, 10, 0)    
    mapdl.rmodif(3, 12, 0)    
    mapdl.rmodif(3, 36 , 167) 
    mapdl.r(3,"")
    mapdl.rmore("")
    mapdl.rmore("", 0)
    mapdl.rmore(0)

    mapdl.et(6, "SURF154")
    mapdl.keyopt(6, 7, 0)
    mapdl.toffst(273.15)
    mapdl.tref(22)

    # Defining material properties
    mapdl.mp("DENS", 1, density) 
    mapdl.tb("HYPE", 1, 1, 3, "YEOH")
    mapdl.tbdata(1,
                yeoh_c1,
                yeoh_c2,
                yeoh_c3,
                yeoh_d1,
                yeoh_d2,
                yeoh_d3)

    # Generate
    mapdl.vmesh("all")

    # Fixed boundary condition
    mapdl.nsel("S", "LOC", "Y", 0)
    mapdl.d("ALL", "ALL")

    mapdl.nsel("S", "LOC", "Y", length * mm2m)
    mapdl.nsel("R", "LOC", "Z", 0, node_tolerance)
    tip_node_index = mapdl.queries.node((width / 2) * mm2m, length * mm2m, 0) - 1

    # Contact region (OuterWalls)
    for i in range(n_chambers - 1):
        mapdl.nsel("S", "LOC", "Y", (i * (chamber_length + slit_length) + end_thickness + chamber_length - slit_thickness) * mm2m)
        mapdl.nsel("R", "LOC", "Z", (bottom_thickness + connecting_chamber_height + slit_bottom_thickness) * mm2m,
                                    height * mm2m)
        mapdl.esln("S", 0)
        mapdl.type(2)
        mapdl.esurf()
        mapdl.type(3)
        mapdl.esurf()

        mapdl.nsel("S", "LOC", "Y", (i * (chamber_length + slit_length) + end_thickness + chamber_length - slit_thickness + slit_length) * mm2m)
        mapdl.nsel("R", "LOC", "Z", (bottom_thickness + connecting_chamber_height + slit_bottom_thickness) * mm2m,
                                    height * mm2m)
        mapdl.esln("S", 0)
        mapdl.type(2)
        mapdl.esurf()
        mapdl.type(3)
        mapdl.esurf()

    # Bottom surface of inner chamber
    mapdl.nsel("S", "LOC", "X", side_thickness * mm2m,
                                (width - side_thickness) * mm2m)
    mapdl.nsel("R", "LOC", "Y", end_thickness * mm2m,
                                (length - end_thickness) * mm2m)
    mapdl.nsel("R", "LOC", "Z", bottom_thickness * mm2m)
    mapdl.esln("S", 0)
    mapdl.nsel("R", "EXT") # Retain only external nodes
    mapdl.esln("S", 0)
    mapdl.type(6)
    mapdl.mat(6)
    mapdl.esurf()
    
    # Generate surface elements on all pressure-loaded chamber faces
    for i in range(n_chambers): 
        # End faces
        mapdl.nsel("S", "LOC", "X", side_thickness * mm2m,
                                    (width - side_thickness) * mm2m)
        mapdl.nsel("R", "LOC", "Y", (i * (chamber_length + slit_length) + end_thickness) * mm2m)
        mapdl.nsel("R", "LOC", "Z", bottom_thickness * mm2m,
                                    (bottom_thickness + chamber_height) * mm2m)
        mapdl.esln("S", 0)
        mapdl.type(6)
        mapdl.mat(6)
        mapdl.esurf()
        
        mapdl.nsel("S", "LOC", "X", side_thickness * mm2m,
                                    (width - side_thickness) * mm2m)
        mapdl.nsel("R", "LOC", "Y", (i * (chamber_length + slit_length) + end_thickness + chamber_length - 2 * slit_thickness) * mm2m)
        mapdl.nsel("R", "LOC", "Z", bottom_thickness * mm2m,
                                    (bottom_thickness + chamber_height) * mm2m)
        mapdl.esln("S", 0)
        mapdl.type(6)
        mapdl.mat(6)
        mapdl.esurf()
        
        # Side faces
        mapdl.nsel("S", "LOC", "X", side_thickness * mm2m)
        mapdl.nsel("R", "LOC", "Y", (i * (chamber_length + slit_length) + end_thickness) * mm2m,
                                    (i * (chamber_length + slit_length)  +end_thickness + chamber_length - 2 * slit_thickness) * mm2m)
        mapdl.nsel("R", "LOC", "Z", bottom_thickness * mm2m,
                                    (bottom_thickness + chamber_height) * mm2m)
        mapdl.esln("S", 0)
        mapdl.type(6)
        mapdl.mat(6)
        mapdl.esurf()
        
        mapdl.nsel("S", "LOC", "X", (width - side_thickness) * mm2m)
        mapdl.nsel("R", "LOC", "Y", (i * (chamber_length + slit_length) + end_thickness) * mm2m,
                                    (i * (chamber_length + slit_length) + end_thickness + chamber_length - 2 * slit_thickness) * mm2m)
        mapdl.nsel("R", "LOC", "Z", bottom_thickness * mm2m, (bottom_thickness + chamber_height) * mm2m)
        mapdl.esln("S", 0)
        mapdl.type(6)
        mapdl.mat(6)
        mapdl.esurf()
        
        # Top faces
        mapdl.nsel("S", "LOC", "X", side_thickness * mm2m, (width - side_thickness) * mm2m)
        mapdl.nsel("R", "LOC", "Y", (i * (chamber_length + slit_length) + end_thickness) * mm2m,
                                    (i * (chamber_length + slit_length) + end_thickness + chamber_length - 2 * slit_thickness) * mm2m)
        mapdl.nsel("R", "LOC", "Z", (bottom_thickness + chamber_height) * mm2m)
        mapdl.esln("S", 0)
        mapdl.type(6)
        mapdl.mat(6)
        mapdl.esurf()

        if i != (n_chambers - 1):            
            
            # Top of connecting chamber
            mapdl.nsel("S", "LOC", "X", (width / 2 - connecting_chamber_width_top / 2) * mm2m, (width / 2 + connecting_chamber_width_top / 2) * mm2m)
            mapdl.nsel("R", "LOC", "Y", (i * (chamber_length + slit_length) + end_thickness + chamber_length - 2 * slit_thickness) * mm2m,
                                        (i * (chamber_length + slit_length) + end_thickness + chamber_length + slit_length) * mm2m)
            mapdl.nsel("R", "LOC", "Z", (bottom_thickness + connecting_chamber_height) * mm2m)
            mapdl.esln("S", 0)
            mapdl.type(6)
            mapdl.mat(6)
            mapdl.esurf()      
            
            connecting_chamber_triangle_adjacent = (connecting_chamber_width_bottom - connecting_chamber_width_top) / 2
            connecting_chamber_angle = math.degrees(
                math.atan(connecting_chamber_height / connecting_chamber_triangle_adjacent)
            )
            hypotenuse = math.sqrt(connecting_chamber_triangle_adjacent ** 2 + connecting_chamber_height ** 2)
            
            # angled connecting chamber walls
            mapdl.local(kcn = 11,
                        kcs = "CART",
                        xc = (width / 2 + connecting_chamber_width_bottom / 2) * mm2m,
                        yc = 0,
                        zc = bottom_thickness * mm2m,
                        thzx = connecting_chamber_angle
            )
            mapdl.csys(11)
            mapdl.nsel("S", "LOC", "X", 0, -hypotenuse)
            mapdl.nsel("R", "LOC", "Y", (i * (chamber_length + slit_length) + end_thickness + chamber_length - 2 * slit_thickness) * mm2m,
                                        (i * (chamber_length + slit_length) + end_thickness + chamber_length + slit_length) * mm2m)
            mapdl.nsel("R", "LOC", "Z", 0)
            mapdl.esln("S", 0)
            mapdl.type(6)
            mapdl.mat(6)
            mapdl.esurf()
            
            mapdl.csys(0)
        
            mapdl.local(kcn = 12,
                        kcs = "CART",
                        xc = (width / 2 - connecting_chamber_width_bottom / 2) * mm2m,
                        yc = 0,
                        zc = bottom_thickness * mm2m,
                        thzx = -connecting_chamber_angle
            )
            mapdl.csys(12)
            mapdl.nsel("S", "LOC", "X", 0, hypotenuse)
            mapdl.nsel("R", "LOC", "Y", (i * (chamber_length + slit_length) + end_thickness + chamber_length - 2 * slit_thickness) * mm2m,
                                        (i * (chamber_length + slit_length) + end_thickness + chamber_length + slit_length) * mm2m)
            mapdl.nsel("R", "LOC", "Z", 0)
            mapdl.esln("S", 0)
            mapdl.type(6)
            mapdl.mat(6)
            mapdl.esurf()
            
            mapdl.csys(0)

    # Apply pressure to internal chamber surfaces
    mapdl.esel("S", "TYPE", "", 6)
    mapdl.nsle()
    mapdl.sf("ALL", "PRES", pressure)

    mapdl.nsel("ALL")
    mapdl.esel("ALL")

    # Gravity
    mapdl.acel(0,9.80665,0)

    # Nonlinear static solution
    mapdl.run("/SOL")
    mapdl.antype(0) # Static analysis 
    mapdl.nlgeom("ON") # Large deformation on
    mapdl.eqslv("SPARSE", "", 1)
    mapdl.deltim(0.1, 0.0004, 1, "OFF") # Initial time step, minimum time step, final time
    mapdl.autots(1) # Automatic time stepping
    mapdl.time(1) # Final time
    mapdl.outres("ALL", "ALL")
    mapdl.solve()
    mapdl.finish()

    result = mapdl.result
    
    _, nodal_displacement = result.nodal_solution(-1)
    tip_vertical_displacement = abs(nodal_displacement[tip_node_index][2])
    tip_horizontal_displacement = length * mm2m + nodal_displacement[tip_node_index][1]
    if tip_horizontal_displacement < 0:
        theta = 90 + 180 / math.pi * (math.atan(-tip_vertical_displacement / tip_horizontal_displacement))
    else:
        theta = 180 / math.pi*(math.atan(tip_vertical_displacement / tip_horizontal_displacement))

    if use_gprc:
        x = np.array([[chamber_height, slit_thickness, slit_length]])
        x_scaled = scaler.transform(x)
        residual = gp.predict(x_scaled, return_std=False)
        residual = residual[0]
        theta += residual

    return theta