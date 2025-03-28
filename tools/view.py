import sys

sys.argv.pop(0)
if sys.argv:
    xdmf_path = sys.argv.pop(0)
    png_path = sys.argv.pop(0)
else:
    xdmf_path = "vel.000000002.xdmf2"
    png_path = "vel.000000002.png"
import paraview
from paraview.simple import *

view = CreateView("RenderView")
view.OrientationAxesVisibility = 0
view.CameraParallelProjection = 1
view.Background = 1, 1, 1
layout = CreateLayout()
layout.AssignView(0, view)
xdmf = OpenDataFile(xdmf_path)
threshold = Threshold(Input=xdmf)
threshold.Scalars = 'CELLS', 'chi'
threshold.UpperThreshold = 0.5
disp = Show(threshold)
tmpLUT = GetColorTransferFunction('tmp')
tmpLUT.RGBPoints = [-1.0, 0.176471, 0.0, 0.294118, -0.87451, 0.272434, 0.095963, 0.444214, -0.74902, 0.373395, 0.228912, 0.56932, -0.623529, 0.481661, 0.415917, 0.657901, -0.498039, 0.601922, 0.562937, 0.750481, -0.372549, 0.718493, 0.695886, 0.836986, -0.24705900000000003, 0.811995, 0.811534, 0.898501, -0.12156900000000004, 0.894733, 0.8995, 0.940023, 0.00392156999999993, 0.969166, 0.966859, 0.963629, 0.12941200000000008, 0.98639, 0.910265, 0.803691, 0.25490199999999996, 0.995002, 0.835371, 0.624375, 0.38039200000000006, 0.992541, 0.736947, 0.420146, 0.5058820000000002, 0.931949, 0.609458, 0.224221, 0.631373, 0.85075, 0.483968, 0.069819, 0.7568630000000001, 0.740023, 0.380623, 0.035371, 0.8823530000000002, 0.617993, 0.29827, 0.026759, 1.0, 0.498039, 0.231373, 0.031373]
disp.LookupTable = tmpLUT
disp.Representation = 'Surface'
disp.ColorArrayName = ['CELLS', 'tmp']

view.CameraParallelProjection = 1
view.UseColorPaletteForBackground = 0
view.Background = 1, 1, 1

xl, xy, yl, yh, zl, zh = threshold.GetDataInformation().GetBounds()
margin = 0
cx = 0.5 * (xl + xy)
cy = 0.5 * (yl + yh)
cz = 0.5 * (zl + zh)
dx = 0.5 * (xh - xl) * (1 + margin)
dy = 0.5 * (yh - yl) * (1 + margin)
dz = 0.5 * (zh - zl) * (1 + margin)

width = 1200
height = int(width * dy / dx)
view.ViewSize = width, height
layout.SetSize(width, height)

view.CameraFocalPoint = cx, cy, cz
view.CameraPosition = cx, cy, cz + 2 * max(dx, dy, dz)
view.CameraParallelScale = max(dx, dy) // 2
SaveScreenshot(png_path, view)
