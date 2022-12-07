# FIXME This is a test implementation of SkewedAxes. It has things to do.
# * We need overload xlim and ylim.
# * Also, we need to support a mode where axes diamond is fixed and the
#    transSkew is adjusted for the change in xlim and ylim.
# * We also need a way to automatically adjust the ha and va of ticklabels
#   and axislabel.
# * Make it compatible w/ the new api.
# * Tick origientation of normal does not work.

# We absorb scaling inside the transSkew, and offset in in transAxes.
# This is necessary as we use aspect=1.
# We may need to implmement Square axes (regardless of aspect), and use
# transAxes for both scaling and translation (Can we really do this?)

from matplotlib import _api
from matplotlib import transforms
from matplotlib.path import Path
from mpl_toolkits.axisartist.axislines import AxisArtistHelper

from matplotlib.transforms import Affine2D

class SkewTransform(Affine2D):
    def __init__(self, rx, ry, sx=1, sy=1):
        super().__init__()
        self._rx, self._ry = rx, ry
        # self._sx, self._sy = sx, sy

        self.update_scale(sx, sy)

    @staticmethod
    def _get_skew_parameters(rx, ry, sx=1, sy=1):
        # get Affined transform from a skewed box defined by (ry, r2) to the
        # transAxes.

        # rx, ry = 0.5, 0.5

        sx0 = 1 - rx
        sy0 = 1 - ry

        import numpy as np
        deg_y = np.degrees(np.arctan2(ry, sx0))
        deg_x = np.degrees(np.arctan2(rx, sy0))

        return sx0, sy0, deg_x, deg_y, rx, 0

    def update_scale(self, sx=None, sy=None):
        self._sx = sx = sx or self._sx
        self._sy = sy = sy or self._sy

        sx0, sy0, deg_x, deg_y, rx, ry = self._get_skew_parameters(self._rx,
                                                                   self._ry,
                                                                   sx, sy)

        self.clear()
        self.scale(sx0/sx, sy0/sy).skew_deg(-deg_x, deg_y).translate(rx, ry)

    def update_r(self, aspect, sx=1, sy=1):
        self._sx, self._sy = sx, sy
        sx0, sy0, deg_x, deg_y, rx, ry = self._get_skew_parameters(self._rx,
                                                                   self._ry)

        self.clear()
        self.scale(sx0/sx, sy0/sy).skew_deg(-deg_x, deg_y).translate(rx, ry)

def get_skew_transform(rx, ry, sx=1, sy=1):
    # get Affined transform from a skewed box defined by (ry, r2) to the
    # transAxes.

    # rx, ry = 0.5, 0.5

    # sx0 = 1 - rx
    # sy0 = 1 - ry

    # import numpy as np
    # deg_y = np.degrees(np.arctan2(ry, (1-rx)))
    # deg_x = np.degrees(np.arctan2(rx, (1-ry)))

    # tr = Affine2D().scale(sx0/sx, sy0/sy).skew_deg(-deg_x, deg_y).translate(rx, 0)

    tr = SkewTransform(rx, ry, sx=sx, sy=sy)
    return tr

import numpy as np
from mpl_toolkits.axisartist.grid_finder import (GridFinder as _GridFinder,
                                                 _find_line_box_crossings)
# def _find_line_box_crossings2(xys, bbox, tr):
#     xys2 = tr.inverted().transform(xys)
#     crossings = _find_line_box_crossings(xys2, bbox)
#     return crossings

def _find_line_box_crossings2(xys0, bbox0, tr0):
    # FIXME This is a modified version of _find_line_box_crossings, which is
    # intended for a skewed bbox defined by the transform tr (from transAxes to
    # the skewed box.)

    import matplotlib.transforms as mtransforms

    tr = tr0 + mtransforms.BboxTransformTo(bbox0) # + tr0
    # tr = Affine2D()
    bbox = mtransforms.Bbox.from_extents(0, 0, 1, 1)
    # bbox = bbox0.transformed(tr)

    # tr = mtransfrom.from_bbox
    xys = tr.inverted().transform(xys0)

    crossings = []
    dxys0 = xys0[1:] - xys0[:-1]
    dxys = xys[1:] - xys[:-1]
    for sl in [slice(None), slice(None, None, -1)]:
        us, vs = xys.T[sl]  # "this" coord, "other" coord
        dus, dvs = dxys.T[sl]
        umin, vmin = bbox.min[sl]
        umax, vmax = bbox.max[sl]
        for u0, inside in [(umin, us > umin), (umax, us < umax)]:
            crossings.append([])
            idxs, = (inside[:-1] ^ inside[1:]).nonzero()
            for idx in idxs:
                v = vs[idx] + (u0 - us[idx]) * dvs[idx] / dus[idx]
                if not vmin <= v <= vmax:
                    continue
                crossing = (u0, v)[sl]
                crossing0 = tr.transform_point(crossing)
                theta0 = np.degrees(np.arctan2(*dxys0[idx][::-1]))
                crossings[-1].append((crossing0, theta0))
    return crossings


class GridFinderSkewed(_GridFinder):
    def __init__(self,
                 transform,
                 extreme_finder=None,
                 grid_locator1=None,
                 grid_locator2=None,
                 tick_formatter1=None,
                 tick_formatter2=None,
                 rx=0, ry=None):
        super().__init__(
            transform,
            extreme_finder=extreme_finder,
            grid_locator1=grid_locator1,
            grid_locator2=grid_locator2,
            tick_formatter1=tick_formatter1,
            tick_formatter2=tick_formatter2)

        self._rx = rx
        self._ry = ry
        self.transSkew = get_skew_transform(rx, ry)

    def _clip_grid_lines_and_find_ticks(self, lines, values, levs, bb):

        # It seems that we are doing this in data coordinate of the host axes.
        # This will give you correct results, if the aspect of the host is 1.
        # The results will be incorrect otherwise.

        gi = {
            "values": [],
            "levels": [],
            "tick_levels": dict(left=[], bottom=[], right=[], top=[]),
            "tick_locs": dict(left=[], bottom=[], right=[], top=[]),
            "lines": [],
        }

        tck_levels = gi["tick_levels"]
        tck_locs = gi["tick_locs"]
        for (lx, ly), v, lev in zip(lines, values, levs):
            tcks = _find_line_box_crossings2(np.column_stack([lx, ly]), bb,
                                             self.transSkew)
            gi["levels"].append(v)
            gi["lines"].append([(lx, ly)])

            for tck, direction in zip(tcks,
                                      ["left", "right", "bottom", "top"]):
                for t in tck:
                    tck_levels[direction].append(lev)
                    tck_locs[direction].append(t)

        return gi

from mpl_toolkits.axisartist.grid_helper_curvelinear import (
    GridHelperBase,
    GridHelperCurveLinear,
    FixedAxisArtistHelper as _FixedAxisArtistHelper)
from mpl_toolkits.axisartist.axis_artist import AxisArtist

class FixedAxisArtistHelper(_FixedAxisArtistHelper):
    """
    Helper class for a fixed axis.
    """
    def __init__(self, grid_helper, side, nth_coord_ticks=None):

        super().__init__(grid_helper, side,
                         nth_coord_ticks=nth_coord_ticks)
        self.transSkew = grid_helper.transSkew
    # def __init__(self, grid_helper, side, nth_coord_ticks):
    #     super().__init__(grid_helper, side, nth_coord_ticks=nth_coord_ticks)
    #     self.grid_helper = grid_helper

    def get_line(self, axes):
        p = super().get_line(axes)
        # print(p)
        return p

    def get_line_transform(self, axes):
        return  self.transSkew + axes.transAxes

class GridHelperSkewed(GridHelperCurveLinear):

    def new_fixed_axis(self, loc,
                       nth_coord=None,
                       axis_direction=None,
                       offset=None,
                       axes=None):
        # if axes is None:
        #     axes = self.axes
        if axis_direction is None:
            axis_direction = loc
        _helper = FixedAxisArtistHelper(self, loc, nth_coord_ticks=nth_coord)
        axisline = AxisArtist(axes, _helper, axis_direction=axis_direction)
        # Why is clip not set on axisline, unlike in new_floating_axis or in
        # the floating_axig.GridHelperCurveLinear subclass?
        return axisline

    def __init__(self, aux_trans,
                 extreme_finder=None,
                 grid_locator1=None,
                 grid_locator2=None,
                 tick_formatter1=None,
                 tick_formatter2=None,
                 rx=0., ry=None):
        """
        aux_trans : a transform from the source (curved) coordinate to
        target (rectilinear) coordinate. An instance of MPL's Transform
        (inverse transform should be defined) or a tuple of two callable
        objects which defines the transform and its inverse. The callables
        need take two arguments of array of source coordinates and
        should return two target coordinates.

        e.g., ``x2, y2 = trans(x1, y1)``
        """
        GridHelperBase.__init__(self)
        # super().__init__()
        self._grid_info = None
        self._aux_trans = aux_trans
        self._rx, self._ry = rx, ry
        self.grid_finder = GridFinderSkewed(aux_trans,
                                            extreme_finder,
                                            grid_locator1,
                                            grid_locator2,
                                            tick_formatter1,
                                            tick_formatter2,
                                            rx=rx, ry=ry)

        self.transSkew = self.grid_finder.transSkew

    def update_lim(self, axes):
        # FIXME We may need to update this so that update_grid is done in
        # display coordinates.
        x1, x2 = axes.get_xlim()
        y1, y2 = axes.get_ylim()
        tr_modified = True  # FIXME we need to check if _aux_trans has been
                            # modified.
        if self._old_limits != (x1, x2, y1, y2) or tr_modified:
            self._update_grid(x1, y1, x2, y2)
            self._old_limits = (x1, x2, y1, y2)

    def _update_grid(self, x1, y1, x2, y2):
        self._grid_info = self.grid_finder.get_grid_info(x1, y1, x2, y2)


from mpl_toolkits.axisartist import Axes as _Axes

class SkewedAxes(_Axes):
    def set_xlim(self, *kl, **kwargs):
        super().set_xlim(*kl, **kwargs)


if True:
    from mpl_toolkits.axisartist.grid_helper_curvelinear import (
        GridHelperCurveLinear)

    import matplotlib.pyplot as plt
    # def tr(x, y): return x, y - x
    # def inv_tr(x, y): return x, y + x

    rx, ry = 0.5, 0.5 # for transform
    rx1, ry2 = 0.8, 0.2 # for axes area
    tr = get_skew_transform(rx, ry, sx=2, sy=1)
    # inv_tr = tr.inverted()
    # tr = Affine2D()

    # grid_helper = GridHelperCurveLinear((tr, inv_tr))
    # grid_helper = GridHelperSkewed(mtransforms.Affine2D(), rx=rx, ry=ry)
    grid_helper = GridHelperSkewed(tr, rx=rx1, ry=ry2)

    fig = plt.figure(2)
    fig.clf()

    from mpl_toolkits.axes_grid1.parasite_axes import (
        host_axes_class_factory, parasite_axes_class_factory)
    HostAxes = SubplotHost = host_axes_class_factory(SkewedAxes)

    ax1 = fig.add_subplot(1, 1, 1, axes_class=HostAxes, grid_helper=grid_helper)
    ax1.set_aspect(1)
    ax1.grid(True)
    ax1.axis[:].major_ticks.set_tickdir("out")
    ax1.axis[:].major_ticks.set_tick_orientation("parallel")

    # ax1.set_ylim(0, 2)
    tr.update_scale(1, 1)
    plt.draw()

    ax2 = ax1.get_aux_axes(tr)
    # ax2.plot([0, 1, 2, 3], [0, 1, 2, 3], "o")

    plt.show()
