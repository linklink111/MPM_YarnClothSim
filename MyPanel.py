import bpy
import time
import numpy as np
import bmesh
from bpy_extras.io_utils import ImportHelper 
from bpy.types import Operator

# https://gitee.com/pampa666/blender_ui
StitchMesh = bpy.data.texts["StitchMesh"].as_module()
CurveAnime = bpy.data.texts["Load_curve_animation"].as_module()
WoveMesh = bpy.data.texts["Wove_Mesh"].as_module()
LoadBCC = bpy.data.texts["loadBCC"].as_module()

class PT_MyUI_Panel(bpy.types.Panel):
    bl_idname = 'MT_MYUI_PT_my_panel'
    bl_label = 'my addon label'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TestUI'

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.label(text='this is a label')
        
        col = layout.column()
        col.prop(context.scene.my_properties, 'blbl_string1', text="str1")


        col.prop(context.scene.my_properties, 'blbl_float1', text="float1")
        col.prop(context.scene.my_properties, 'blbl_int1', text="int1")
        col.prop(context.scene.my_properties, 'blbl_enum1', text="enum1")
        col.prop(context.scene.my_properties, 'blbl_int2', text="rotate")
        row1 = layout.row()
        row1.operator("my.blblop1", text='create stitch for face', icon='LIGHT')
        row2 = layout.row()
        row2.operator("my.create_stitch_for_object", text='create stitch for obj', icon='LIGHT')
        row3 = layout.row()
        row3.operator("my.create_stitch_for_column", text='create_stitch_column', icon='LIGHT')
        
        row3_ = layout.row()
        row3_.operator("my.open_filebrowser", text='load BCC file', icon='LIGHT')
        col_ = layout.column()
        col_.prop(context.scene.my_properties, 'blbl_float3', text="length")
        row3__ = layout.row()
        row3__.operator("my.select_long_edges", text='select long edges', icon='LIGHT')
        
        row4 = layout.row()
        row4.label(text='key frames')
        row5 = layout.row()
        row5.operator("my.delete_key_frames", text='delete all key frames', icon='LIGHT')
        
        col1 = layout.column()
        col1.prop(context.scene.my_properties, 'blbl_int3', text="start")
        col1.prop(context.scene.my_properties, 'blbl_int4', text="end")
        col1.prop(context.scene.my_properties, 'blbl_int41', text="skip")
        row6 = layout.row()
        row6.operator("my.load_key_frames", text='load key frames', icon='LIGHT')
        row7 = layout.row()
        row7.label(text='Woven cloth')
        col2 = layout.column()
        col2.prop(context.scene.my_properties, 'blbl_int5', text="w")
        col2.prop(context.scene.my_properties, 'blbl_int6', text="h")
        row8 = layout.row()
        row8.operator("my.create_wove", text='wove face', icon='LIGHT')
        col3 = layout.column()
        col3.prop(context.scene.my_properties, 'blbl_int7', text="direction")
        row8_1 = layout.row()
        row8_1.operator("my.create_wove_f", text='wove one face', icon='LIGHT')





class OT_My_Operator(bpy.types.Operator):
    '''Operator info'''
    bl_idname = 'my.blblop1'
    bl_label = 'My Operator'
    
    def execute(self, context):
        my_properties = context.scene.my_properties
        StitchMesh.convert_selected_faces_to_stitch_mesh(rot=my_properties.blbl_int2)
        return {'FINISHED'}
    
class OT_Create_Stitch(bpy.types.Operator):
    bl_idname = 'my.create_stitch_for_object'
    bl_label = 'create_stitch_operator'
    
    def execute(self, context):
        name = bpy.context.object.name # active object here
        StitchMesh.convert_mesh_to_stitch_mesh(name)
        return {'FINISHED'}

class OT_Create_Stitch_In_Column(bpy.types.Operator):
    bl_idname = 'my.create_stitch_for_column'
    bl_label = 'create_stitch_operator'
    
    def execute(self, context):
        StitchMesh.iterate_face_cloumn_and_create_stitch()
        return {'FINISHED'}

class OT_Load_Key_Frames(bpy.types.Operator):
    bl_idname = 'my.load_key_frames'
    bl_label = 'load_key_frames'
    
    def execute(self, context):
        my_properties = context.scene.my_properties
        start = my_properties.blbl_int3
        end = my_properties.blbl_int4
        skip = my_properties.blbl_int41
        CurveAnime.load_animation_to_shape_key(start,end,skip=skip)
        CurveAnime.animate_shape_keys(start,end)
        return {'FINISHED'}

class OT_Del_Key_Frames(bpy.types.Operator):
    bl_idname = 'my.delete_key_frames'
    bl_label = 'delete_key_frames'
    
    def execute(self, context):
        CurveAnime.delete_all_shape_keys()
        return {'FINISHED'}

class OT_Create_Wove(bpy.types.Operator):
    bl_idname = 'my.create_wove'
    bl_label = 'create woven cloth'
    
    def execute(self, context):
        my_properties = context.scene.my_properties
        w = my_properties.blbl_int5
        h = my_properties.blbl_int6
        WoveMesh.create_woven_fabric(w,h)
        return {'FINISHED'}
class OT_Create_Wove_On_Face(bpy.types.Operator):
    bl_idname = 'my.create_wove_f'
    bl_label = 'create wove on_face'
    
    def execute(self, context):
        my_properties = context.scene.my_properties
        dir = my_properties.blbl_int7
        WoveMesh.wove_on_face(dir)
        return {'FINISHED'}
    
    
class OT_TestOpenFilebrowser(Operator, ImportHelper): 
    bl_idname = "my.open_filebrowser" 
    bl_label = "Open the file browser (yay)" 
    def execute(self, context): 
        """Do something with the selected file(s).""" 
        LoadBCC.load_bcc_file(self.filepath)
        return {'FINISHED'}
    
class OT_SelectLongEdges(Operator): 
    bl_idname = "my.select_long_edges" 
    bl_label = "Select long edges" 
    def execute(self, context): 
        my_properties = context.scene.my_properties
        length = my_properties.blbl_float3
        LoadBCC.select_long_edges(length)
        return {'FINISHED'}

class My_Properties(bpy.types.PropertyGroup):
    blbl_string1: bpy.props.StringProperty(name="blbl_string1")
    blbl_string2: bpy.props.StringProperty(name="audio_fname")


    blbl_int1: bpy.props.IntProperty(name="blbl_int1")
    
    blbl_int2: bpy.props.IntProperty(name="rotate")
    blbl_int3: bpy.props.IntProperty(name="start")
    blbl_int4: bpy.props.IntProperty(name="end")
    blbl_int41: bpy.props.IntProperty(name="skip",default=1)
    
    blbl_int5: bpy.props.IntProperty(name="w")
    blbl_int6: bpy.props.IntProperty(name="h")
    
    blbl_int7: bpy.props.IntProperty(name="dir")
    
    blbl_int8: bpy.props.IntProperty(name="high_cut")
    blbl_int9: bpy.props.IntProperty(name="low_cut")

    blbl_float1: bpy.props.FloatProperty(name="blbl_float1", soft_min=-1.0, soft_max=1.0)
    blbl_float2: bpy.props.FloatProperty(name="blbl_float2", soft_min=-1.0, soft_max=100.0)
    blbl_float3: bpy.props.FloatProperty(name="blbl_float3", soft_min=0.0, soft_max=100.0)

    blbl_enum1: bpy.props.EnumProperty(
        name="blbl_enum1",
        description='this is option',
        items=[
            ('OP1', 'this is op1', 'help info about op1'),
            ('OP2', 'this is op2', 'help info about op2'),
            ('OP3', 'this is op3', 'help info about op3')]
        )


bpy.utils.register_class(OT_My_Operator)
bpy.utils.register_class(OT_Create_Stitch)
bpy.utils.register_class(OT_Create_Stitch_In_Column)
bpy.utils.register_class(PT_MyUI_Panel)
bpy.utils.register_class(My_Properties)
bpy.utils.register_class(OT_Load_Key_Frames)
bpy.utils.register_class(OT_Del_Key_Frames)
bpy.utils.register_class(OT_Create_Wove)
bpy.utils.register_class(OT_Create_Wove_On_Face)
bpy.utils.register_class(OT_TestOpenFilebrowser)
bpy.utils.register_class(OT_SelectLongEdges)

bpy.types.Scene.my_properties = bpy.props.PointerProperty(type=My_Properties)