# MPM_YarnClothSim
MPM yarn based cloth simulation

欢迎来玩 MPM yarn based cloth simulation， 这是基于蒋陈凡夫老师的 Anisotropic elastoplasicity for cloth, knit and hair frictional contact 写的 模拟 yarn based cloth 的程序（目前只能模拟yarn和yarn based cloth，不能模拟沙子或三角网格布料orz），参考了知乎橘子猫大佬的2D版本（https://zhuanlan.zhihu.com/p/414356129）
由于本人很菜QAQ，程序目前存在如下问题：
- return mapping好像实现错了，所以在模拟中没有执行这个步骤；
- 大多数时候布料在一个方向上会无限收缩，知道成为一条线；
- 出现其他奇怪的现象。

如果您能帮助我改进，我将万分感谢！

## Usage

运行方法：

- 用 Blender2.93 打开 YarnSound.bend（主要的模拟程序是 UnitTest_AEPMPM.py (由于改文件名会发生错误，所以我没有改QAQ））；
- 安装 taichi-Blend 插件；
- 切换到 MyPanel 文本并运行（此时视图中出现交互界面）；

创建yarn based cloth：

- 针织：创建物体->细分->create Stitch for Obj (如果面的index是按顺序的，可以在物体上直接转换，如果面的index是乱序的，可以在编辑模式下转换每一个面)
- 编织：创建物体->细分->wove face
- 导入：交互界面的load BCC file 按钮。

进行模拟：

- 在 Blender 2.93 目录(也就是blender.exe所在的目录)下创建路径
  - work_dir2/frames
- 钉固顶点：编辑曲线并选中要钉固的顶点(如果没有钉固顶点，需要创建一个浮空的顶点并选中它)
- 球体/地面碰撞：在GridCollision中修改球体的位置和半径或地面高度
- 运行：切换到UnitTest_AEPMEM,点击运行
- 运行完成后导入动画：点击 load Animation 按钮 (动画在运行的过程中会写到 work_dir2/frames 下，点击此按钮后保存)
