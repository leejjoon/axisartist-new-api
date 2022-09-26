import numpy as np

from mpl_toolkits.axes_grid1 import mpl_axes
# from mpl_toolkits.axisartist.axislines import (Axes as _Axes,
#                                                GridHelperRectlinear,
#                                                GridHelperCurveLinear)
from mpl_toolkits.axisartist.axis_artist import AxisArtist
from mpl_toolkits.axisartist import (
    Axes as _Axes,
    GridHelperRectlinear,
    GridHelperCurveLinear as _GridHelperCurveLinear
)

from mpl_toolkits.axisartist.grid_helper_curvelinear import (
    FloatingAxisArtistHelper as _FloatingAxisArtistHelper
)

# For the floating axis, the angles for ticks and labels are measured in the
# pixel coordinates. Thus correct values are returned even if aspect is not 1.
# This is not the case for fixed axis.


def measure_local_grid_angles():
    pass

class FloatingAxisArtistHelper(_FloatingAxisArtistHelper):
    def _filter_levels(self, v_min, v_max, levs, factor):
        # we update the min, max and levels using the local extremes.
        e_min, e_max = self._extremes  # ranges of other coordinates

        v_min = max(e_min, v_min)
        v_max = min(e_max, v_max)
        levs = [l for l in levs if v_min <= l/factor <= v_max]

        return v_min, v_max, levs

    def update_lim(self, axes):
        """When the extreme value is specified, the upstream helper uses that extreme
        values to calculate grid info, but still using the locator params from
        the upstream, resulting in that the ticks floating axis can be
        different from those of fixed axis. Being different could be good, but
        then we need to have a control over it.

        * Let's make its ticks identical to those of fixed ones
        * If you want them different, make another grid-helper.

        """
        # FIXME I think this code could be used with the original
        # _FloatingAxisArtistHelper with _filter_leves method returning the
        # inputs.
        self.grid_helper.update_lim(axes)

        x1, x2 = axes.get_xlim()
        y1, y2 = axes.get_ylim()
        grid_finder = self.grid_helper.grid_finder
        extremes = grid_finder.extreme_finder(grid_finder.inv_transform_xy,
                                              x1, y1, x2, y2)

        lon_min, lon_max, lat_min, lat_max = extremes

        # we calculate tick location with original extreme values.
        # nth_coord = 0 means axis along the y axis (fixed x)
        # nth_coord = 1 means axis along the x axis (fixed y)
        levs_n_factor = {
            1: grid_finder.grid_locator1(lon_min, lon_max),
            0: grid_finder.grid_locator2(lat_min, lat_max)
        }

        # we update the min, max and levels using the local extremes.
        v_min, v_max = [(lat_min, lat_max), (lon_min, lon_max)][self.nth_coord]
        levs, n, factor = levs_n_factor[self.nth_coord]
        # Note that actual levels are levs/factor.

        v_min, v_max, levs = self._filter_levels(v_min, v_max, levs, factor)
        levs_n_factor[self.nth_coord] = (levs, n, factor)

        # calculate path of the axis.
        ii0 = np.full(self._line_num_points, self.value)
        jj0 = np.linspace(v_min, v_max, self._line_num_points)
        if self.nth_coord == 0:
            xx, yy = grid_finder.transform_xy(ii0, jj0)
        elif self.nth_coord == 1:
            xx, yy = grid_finder.transform_xy(jj0, ii0)

        lon_levs, _, lon_factor = levs_n_factor[1]
        lat_levs, _, lat_factor = levs_n_factor[0]
        lon_labels = grid_finder.tick_formatter1("bottom", lon_factor, lon_levs)
        lat_labels = grid_finder.tick_formatter2("left", lat_factor, lat_levs)

        self._grid_info = {
            "extremes": (lon_min, lon_max, lat_min, lat_max),
            "lon_info": levs_n_factor[1],
            "lat_info": levs_n_factor[0],
            "lon_labels": lon_labels,
            "lat_labels": lat_labels,
            "line_xy": (xx, yy),
        }

        # print("update_grid_info")

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label, (optionally) tick_label"""
        # We start a copy of same method from axisartist.
        # print("get_tick")
        # self.update_lim(axes)

        grid_finder = self.grid_helper.grid_finder

        # regardless of the nth_coord, we need to have both dx and dy values.
        lat_levs, _, lat_factor = self._grid_info["lat_info"]
        dy = 0.01 / lat_factor

        lon_levs, _, lon_factor = self._grid_info["lon_info"]
        dx = 0.01 / lon_factor

        e1 = max(self._extremes)

        # update_lim method should have filtered the levels already. Thus we
        # don't need to worry about it here.

        if self.nth_coord == 0:
            lat_levs = np.asarray(lat_levs)
            yy0 = lat_levs / lat_factor
            xx0 = np.full_like(yy0, self.value)
        elif self.nth_coord == 1:
            lon_levs = np.asarray(lon_levs)
            xx0 = lon_levs / lon_factor
            yy0 = np.full_like(xx0, self.value)

        def transform_xy(x, y):
            trf = grid_finder.get_transform() + axes.transData
            return trf.transform(np.column_stack([x, y])).T

        # xx1, xx2 are the locations of the ticks in figure coordinates.
        xx1, yy1 = transform_xy(xx0, yy0)

        # find angles
        if self.nth_coord == 0:
            # we modify yy0 so that yy0+dy does not go out of range so that
            # they become ildefined.
            yy0[yy0 + dy > e1] -= dy

            dx1, dy1 = dx, 0
            dx2, dy2 = 0, dy

            labels = self._grid_info["lat_labels"]

        elif self.nth_coord == 1:
            # we modify xx0 so that yy0+dy does not go out of range so that
            # they become ildefined.
            xx0[xx0 + dx > e1] -= dx

            dx1, dy1 = 0, dy
            dx2, dy2 = dx, 0

            labels = self._grid_info["lon_labels"]

        # We try to measure angles for ticks and labels. They are measured in
        # the pixel coordinates, thus correct values are reported even if aspec
        # is not 1.

        # These are reference positions.
        xx2, yy2 = transform_xy(xx0, yy0)

        # to measure angle normal
        xx2a, yy2a = transform_xy(xx0+dx1, yy0+dy1)

        # to measure angle tangent
        xx2b, yy2b = transform_xy(xx0+dx2, yy0+dy2)

        def f1():
            dd = np.arctan2(yy2a-yy2, xx2a-xx2)  # angle normal
            dd2 = np.arctan2(yy2b-yy2, xx2b-xx2)  # angle tangent
            mm = (yy2a == yy2) & (xx2a == xx2)  # mask where dd not defined
            dd[mm] = dd2[mm] + np.pi / 2

            tick_to_axes = self.get_tick_transform(axes) - axes.transAxes
            for x, y, d, d2, lab in zip(xx1, yy1, dd, dd2, labels):
                c2 = tick_to_axes.transform((x, y))
                delta = 0.00001
                if 0-delta <= c2[0] <= 1+delta and 0-delta <= c2[1] <= 1+delta:
                    d1, d2 = np.rad2deg([d, d2])
                    yield [x, y], d1, d2, lab

        return f1(), iter([])


class GridHelperCurveLinear(_GridHelperCurveLinear):
    def new_floating_axis(self, nth_coord,
                          value,
                          axes=None,
                          axis_direction="bottom"
                          ):

        if axes is None:
            axes = self.axes

        _helper = FloatingAxisArtistHelper(
            self, nth_coord, value, axis_direction)

        axisline = AxisArtist(axes, _helper)

        axisline.line.set_clip_on(True)
        axisline.line.set_clip_box(axisline.axes.bbox)

        return axisline

class Axes(_Axes):

    def __init__(self, *args, **kwargs):
        self._axisline_on = False
        super().__init__(*args, **kwargs)
        self._axisline_projections = {}

        self.register_projection("_default_",
                                 None, self.get_grid_helper())

    def register_projection(self, proj_name,
                            transform_to_host, grid_helper=None):

        grid_helper = (grid_helper if grid_helper
                       else GridHelperCurveLinear(transform_to_host))

        self._axisline_projections[proj_name] = dict(
            transform_to_host=transform_to_host,
            grid_helper=grid_helper
        )

    def get_grid_helper(self, proj_name=None):
        if proj_name:
            return self._axisline_projections[proj_name]["grid_helper"]
        else:
            return super().get_grid_helper()

    def get_axisartist_transform(self, proj_name):
        return self._axisline_projections[proj_name]["transform_to_host"]

    def select_visible_projection(self, proj_name):
        self._grid_helper = self.get_grid_helper(proj_name)

        # keep the visiblity of the original axis
        vv = dict((which, axis.get_visible())
                  for which, axis in self.axis.items())
        self._init_axis_artists()
        for which, v in vv.items():
            self.axis[which].set_visible(v)

        # keep the visiblity of the original gridlines
        v = ax.gridlines.get_visible()
        self._init_gridlines()
        self.gridlines.set(visible=v)
        self.gridlines.set_clip_path(self.axes.patch)

        # # self._init_axis_artists()
        # if axes is None:
        #     axes = self

        # self._axislines = mpl_axes.Axes.AxisDict(self)
        # # new_fixed_axis = self.get_grid_helper().new_fixed_axis
        # new_fixed_axis = self.get_grid_helper(proj_name).new_fixed_axis
        # for loc in ["bottom", "top", "left", "right"]:
        #     self._axislines[loc] = new_fixed_axis(loc=loc, axes=axes,
        #                                           axis_direction=loc)

        # for axisline in [self._axislines["top"], self._axislines["right"]]:
        #     axisline.label.set_visible(False)
        #     axisline.major_ticklabels.set_visible(False)
        #     axisline.minor_ticklabels.set_visible(False)


    def cla_old(self):
        # gridlines need to b created before cla() since cla calls grid()
        self._init_gridlines()
        super().cla()

        # the clip_path should be set after Axes.cla() since that's
        # when a patch is created.
        self.gridlines.set_clip_path(self.axes.patch)

        self._init_axis_artists()

    def _init_axis_artists(self, axes=None):
        if axes is None:
            axes = self

        self._axislines = mpl_axes.Axes.AxisDict(self)
        new_fixed_axis = self.get_grid_helper().new_fixed_axis
        for loc in ["bottom", "top", "left", "right"]:
            self._axislines[loc] = new_fixed_axis(loc=loc, axes=axes,
                                                  axis_direction=loc)

        for axisline in [self._axislines["top"], self._axislines["right"]]:
            axisline.label.set_visible(False)
            axisline.major_ticklabels.set_visible(False)
            axisline.minor_ticklabels.set_visible(False)

    # def new_fixed_axis(self, loc,
    #                    nth_coord=None,
    #                    axis_direction=None,
    #                    offset=None,
    #                    axes=None,
    #                    proj_name=None
    #                    ):
    def new_fixed_axis(self, loc, offset=None, proj_name=None):

        # if axes is None:
        #     _api.warn_external(
        #         "'new_fixed_axis' explicitly requires the axes keyword.")
        #     axes = self.axes

        # _helper = AxisArtistHelperRectlinear.Fixed(axes, loc, nth_coord)

        # if axis_direction is None:
        #     axis_direction = loc
        # axisline = AxisArtist(axes, _helper, offset=offset,
        #                       axis_direction=axis_direction,
        #                       )

        if proj_name is None and not isinstance(self, ParasiteAxesBase):
            axis = super().new_fixed_axis(loc,
                                          offset=offset)
        else:
            if isinstance(self, ParasiteAxesBase):
                axes = self._parent_axes
            else:
                axes = self

            gh = self.get_grid_helper(proj_name=proj_name)
            axis = gh.new_fixed_axis(loc,
                                     nth_coord=None,
                                     axis_direction=None,
                                     offset=offset,
                                     axes=axes,
                                     )
        return axis

        # return axisline

    def new_floating_axis(self, nth_coord, value,
                          axis_direction="bottom",
                          proj_name=None,
                          axes=None,
                          ):
        if axes is None:
            if isinstance(self, ParasiteAxesBase):
                axes = self._parent_axes
            else:
                axes = self
        gh = self.get_grid_helper(proj_name)
        axis = gh.new_floating_axis(nth_coord, value,
                                    axis_direction=axis_direction,
                                    axes=axes)
        return axis

    # This may need to be a part of HostAxes
    def get_projected_axes(self, proj_name, axes_class=None):
        # FIXME We should reorganize the class hierachy so that this method is
        # no availble to parasite axes.
        if isinstance(self, ParasiteAxesBase):
            raise RuntimeError("do not call this method from the parasite axes")

        if axes_class is None:
            axes_class = ParasiteAxes

        aux_trans = self.get_axisartist_transform(proj_name)
        # ax = self._add_twin_axes(
        #     axes_class, aux_transform=aux_trans, viewlim_mode="transform")
        ax = self._add_twin_axes(
            axes_class, aux_transform=aux_trans, viewlim_mode="equal")

        # FIXME This is very hacky, but we reassign axis and gridlines. We
        # should have a better way of doing this. Parasite axes should have an
        # attribute to its host axes, and the host axes should be used to
        # created axis and gridlines.
        grid_helper = self.get_grid_helper(proj_name)
        ax._grid_helper = grid_helper

        new_fixed_axis = grid_helper.new_fixed_axis
        for loc in ["bottom", "top", "left", "right"]:
            ax._axislines[loc] = new_fixed_axis(loc=loc, axes=self,
                                                axis_direction=loc)

        ax.gridlines = ax.new_gridlines(grid_helper)
        ax.gridlines.set_transform(self.transData)
        ax.gridlines.set_clip_path(self.axes.patch)
        ax.gridlines.set_visible(False)

        # ax.axis[:].set_visible(False)
        self.axis["top", "right"].set_visible(False)
        ax.axis["top", "right"].set_visible(True)
        ax.axis["left", "bottom"].set_visible(False)
        return ax

from mpl_toolkits.axes_grid1.parasite_axes import (
    ParasiteAxesBase, host_axes_class_factory, parasite_axes_class_factory,
    parasite_axes_auxtrans_class_factory, subplot_class_factory)


# from mpl_toolkits.axes_grid1.parasite_axes import (
#     ParasiteAxesBase as _ParasiteAxesBase
# )

# class ParasiteAxesBase(_ParasiteAxesBase):
#     pass

# from matplotlib import cbook
# parasite_axes_class_factory = cbook._make_class_factory(
#     ParasiteAxesBase, "{}Parasite")

ParasiteAxes = parasite_axes_class_factory(Axes)

HostAxes = host_axes_class_factory(Axes)

if True:
    import matplotlib.pyplot as plt
    plt.clf()
    ax = plt.subplot(111, axes_class=HostAxes, aspect=1)

    from matplotlib.transforms import Affine2D
    # We define transform from the original axes to the rotated one.
    transform_to_rotated = (Affine2D()
                 .translate(0, 0)
                 .rotate_deg(15)
                 .translate(0, 0)
                 .scale(1, 1) # to reduce the height of the histogram
    )

    transform_from_rotated = transform_to_rotated.inverted()

    # We create new parasite axes to draw a x-axis of the rotated axes. Note that
    # the transData of this axes is still that of the original axes, and we cannot
    # use it to draw histogram.


    # helper = AA.GridHelperCurveLinear(transform_from_rotated)
    # ax2_for_axis = AA.ParasiteAxes(ax1, viewlim_mode="equal", grid_helper=helper)
    # ax1.parasites.append(ax2_for_axis)

    proj_name = "rotated"
    ax.register_projection(proj_name,
                           transform_from_rotated, grid_helper=None)

    # ax.add_twin(name="rotaed", projection=None)

    # ax.select_visible_projection(proj_name) # to replace ticks and ticklabels.
    # ax.grid(True)

    ax_rotated = ax.get_projected_axes(proj_name)

    ax.plot([0], [0], "bo", ms=10, mfc="none")
    ax_rotated.plot([0, 0], [0, 1], "ro")

    ax.set(xlim=(0, 3), ylim=(0, 2))

    ax.grid(color="r", alpha=0.2)
    ax_rotated.grid(color="b", alpha=0.2)

    # ax_rotated.axis["right"].set_visible(True)
    # ax_rotated.axis["right"].toggle(all=True)

    # ax_rotated.register_projection(proj_name,
    #                                transform_from_rotated, grid_helper=None)
    # ax_rotated.select_visible_projection(proj_name) # to replace ticks and ticklabel_formatabels.

    # gh = ax.get_grid_helper("rotated")

    # FIXME Please make it work.
    # ax.locator_params(steps=[1., 5, 10], nbins=8)

    gh = ax_rotated.get_grid_helper()
    l = gh.grid_finder.grid_locator1
    l.set_params(steps=[1., 2, 5,10])
    l.set_params(nbins=10)
    l = gh.grid_finder.grid_locator2
    l.set_params(steps=[1., 5, 10])
    l.set_params(nbins=8)

    ax.locator_params(steps=[1., 5, 10], nbins=8)

    # ax.axis["right2"] = ax.new_fixed_axis("right", offset=(30, 0))

    # FIXED axis

    ax_rotated.axis["top", "right"].set_visible(False)

    # Duplicating right and top axis, just for a test.
    # It does not matter which axes the new axis is added to.
    ax.axis["right2"] = ax.new_fixed_axis("right",
                                          proj_name=proj_name)
    # OR
    ax_rotated.axis["top2"] = ax_rotated.new_fixed_axis("top")

    # FLOATING axis

    ax.axis["y=1"] = ax.new_floating_axis(1, 1, "bottom", proj_name="rotated")

    ax.axis["y=1"].get_helper().set_extremes(0, 1) # limit its extents.

    ax_rotated.axis["y=2"] = ax_rotated.new_floating_axis(1, 2,
                                                          "bottom")
    # ax_rotated.axis["y=2"].set_transform(ax.transData)

    # FIXME maybe we introduce axis.set_lim method?
    ax_rotated.axis["y=2"].get_helper().set_extremes(0.5, 2) # limit its extents.

    ax_rotated.axis["x=2"] = ax_rotated.new_floating_axis(0, 2,
                                                          "right")
    ax_rotated.axis["x=2"].get_helper().set_extremes(1, 2) # limit its extents.

    # ax.new_floating_axis(proj_name, nth_coord=0, trail=trail, dir="left")
    # extent = [0, 0, 1, 1]
    # ax_floating = ax.new_floating_axes(proj_name, extent)
