# MPM_YarnClothSim
MPM yarn based cloth simulation


## Usage

运行方法：

- 用 Blender2.93 打开 YarnSound.bend；
- 安装 taichi-Blend 插件；
- 切换到 MyPanel 文本并运行（此时视图中出现交互界面）；

创建纱线布料：

- 针织：创建物体->细分->create Stitch for Obj (如果面的index是按顺序的，可以在物体上直接转换，如果面的index是乱序的，可以在编辑模式下转换每一个面)
- 编织：创建物体->细分->wove face
- 导入：交互界面的load BCC file 按钮。

进行模拟：

- 在 Blender 2.93 目录(也就是blender.exe所在的目录)下创建路径
  - work_dir2/frames
- 钉固顶点：编辑曲线并选中要钉固的顶点(如果没有钉固顶点，需要创建一个浮空的顶点并选中它)
- 球体/地面碰撞：在GridCollision中修改球体的位置和半径或地面高度
- 运行：切换到UnitTest_AEPMEM,点击运行
