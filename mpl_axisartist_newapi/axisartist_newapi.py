import numpy as np

import matplotlib as mpl

from matplotlib import _api
from mpl_toolkits.axes_grid1 import mpl_axes
# from mpl_toolkits.axisartist.axislines import (Axes as _Axes,
#                                                GridHelperRectlinear,
#                                                GridHelperCurveLinear)
from mpl_toolkits.axisartist.axis_artist import (AxisArtist as _AxisArtist)
from mpl_toolkits.axisartist import (
    Axes as _Axes,
    GridHelperRectlinear,
    GridHelperCurveLinear as _GridHelperCurveLinear,
    GridlinesCollection
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
    def __init__(self, *kl, **kw):
        self._extreme_mode = kw.pop("extreme_mode", "host")
        self._ticklabel_angle_mode = kw.pop("ticklabel_angle_mode", "axis")
        super().__init__(*kl, **kw)

    def _get_host_transfrom(self, host):
        return host.transData

    def _filter_levels(self, v_min, v_max, levs, factor):
        # we update the min, max and levels using the local extremes.
        e_min, e_max = self._extremes  # ranges of other coordinates

        v_min = max(e_min, v_min)
        v_max = min(e_max, v_max)
        levs = [l for l in levs if v_min <= l/factor <= v_max]

        return v_min, v_max, levs

    def update_lim(self, axes):
        ""

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

        # The _grid_info object from grid_helper.update_lim is for grids in
        # rectlinear bounding box, sor the calculated information is different
        # from what we need from here, although there is overwrap. We can
        # refactor for better code sharing. But it would be just extreme and
        # levels from the locator.

        # Also, it would be good if different floating axises sharing same
        # grid_helper shares this information.

        # self.grid_helper.update_lim(axes)

        x1, x2 = axes.get_xlim()
        y1, y2 = axes.get_ylim()

        if self._extremes is None or self._extreme_mode == "host":
            # we simply use extreme values found from the axes area of the host
            # axes. This will make usre that tick locations of floating axis
            # will match that of the fixed axis.
            extremes = None
        else:
            if self.nth_coord == 1:
                extremes = list(self._extremes) + [None, None]
            else:
                extremes = [None, None] + list(self._extremes)

        _grid_info = self.grid_helper.get_extreme_tick_locs(x1, x2, y1, y2,
                                                            extremes=extremes)

        lon_min, lon_max, lat_min, lat_max = _grid_info["extremes"]
        grid_finder = self.grid_helper.grid_finder

        # we use tick location with original extreme values.
        # nth_coord = 0 means axis along the y axis (fixed x)
        # nth_coord = 1 means axis along the x axis (fixed y)
        levs_n_factor = {
            1: _grid_info["lon_levs_n_factor"],
            0: _grid_info["lat_levs_n_factor"]
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

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label, (optionally) tick_label"""
        # We start a copy of same method from axisartist.

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
            trf = grid_finder.get_transform() + self._get_host_transfrom(axes)
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
        if self._ticklabel_angle_mode != "axis":
            xx2a, yy2a = transform_xy(xx0-dx1, yy0+dy1)
        else:
            xx2a, yy2a = transform_xy(xx0+dx1, yy0+dy1)

        # to measure angle tangent
        xx2b, yy2b = transform_xy(xx0+dx2, yy0+dy2)

        def f1():
            dd = np.arctan2(yy2a-yy2, xx2a-xx2)  # angle normal
            dd2 = np.arctan2(yy2b-yy2, xx2b-xx2)  # angle tangent
            mm = (yy2a == yy2) & (xx2a == xx2)  # mask where dd not defined

            dd[mm] = dd2[mm] + np.pi / 2

            if self._ticklabel_angle_mode != "axis":
                dd2 = dd - 0.5*np.pi

            tick_to_axes = self.get_tick_transform(axes) - axes.transAxes
            for x, y, d, d2, lab in zip(xx1, yy1, dd, dd2, labels):
                c2 = tick_to_axes.transform((x, y))
                delta = 0.00001
                if 0-delta <= c2[0] <= 1+delta and 0-delta <= c2[1] <= 1+delta:
                    d1, d2 = np.rad2deg([d, d2])
                    yield [x, y], d1, d2, lab

        return f1(), iter([])


class AxisArtist(_AxisArtist):
    def set_lim(self, e1, e2):
        self.get_helper().set_extremes(e1, e2) # limit its extents.


class GridHelperCurveLinear(_GridHelperCurveLinear):

    def get_extreme_tick_locs(self, x1, x2, y1, y2, extremes=None):

        # For now, this will be called from the FloatingAxisHelper, but also
        # could be used from grid_finder.

        # It could beeter if we cache the results as in GridHelper's
        # update_lim.
        grid_finder = self.grid_finder
        _extremes = grid_finder.extreme_finder(grid_finder.inv_transform_xy,
                                               x1, y1, x2, y2)

        if extremes is not None:
            extremes = [e2 if e1 is None or not np.isfinite(e1) else e1
                        for e1, e2 in zip(extremes, _extremes)]
        else:
            extremes = _extremes

        lon_min, lon_max, lat_min, lat_max = extremes

        lon_levs_n_factor = grid_finder.grid_locator1(lon_min, lon_max)
        lat_levs_n_factor = grid_finder.grid_locator2(lat_min, lat_max)

        return dict(extremes=extremes,
                    lon_levs_n_factor=lon_levs_n_factor,
                    lat_levs_n_factor=lat_levs_n_factor)

    def new_floating_axis(self, nth_coord,
                          value,
                          axes=None,
                          axis_direction="bottom",
                          extreme_mode="host",
                          ticklabel_angle_mode="axis",
                          ):

        if axes is None:
            axes = self.axes

        _helper = FloatingAxisArtistHelper(
            self, nth_coord, value, axis_direction,
            extreme_mode=extreme_mode,
            ticklabel_angle_mode=ticklabel_angle_mode,
        )
        # axis_direction parameter to the FloatingAxisArtistHelper has no
        # effct.

        axisline = AxisArtist(axes, _helper, axis_direction=axis_direction)

        axisline.line.set_clip_on(True)
        axisline.line.set_clip_box(axisline.axes.bbox)

        return axisline

    # this method was deprecated in mpl36 and removed in mpl3.7. I simply
    # copied it.
    def new_gridlines(self, ax):
        """
        Create and return a new GridlineCollection instance.

        *which* : "major" or "minor"
        *axis* : "both", "x" or "y"

        """
        gridlines = GridlinesCollection(
            None, transform=ax.transData, colors=mpl.rcParams['grid.color'],
            linestyles=mpl.rcParams['grid.linestyle'],
            linewidths=mpl.rcParams['grid.linewidth'])
        ax._set_artist_props(gridlines)
        gridlines.set_grid_helper(self)

        ax.axes._set_artist_props(gridlines)
        # gridlines.set_clip_path(self.axes.patch)
        # set_clip_path need to be deferred after Axes.cla is completed.
        # It is done inside the cla.

        return gridlines

    def copy(self,
             extreme_finder=None,
             grid_locator1=None,
             grid_locator2=None,
             tick_formatter1=None,
             tick_formatter2=None):

        grid_finder = self.grid_finder

        # if extreme_finder is None:
        #     extreme_finder = grid_finder.extreme_finder
        # if grid_locator1 is None:
        #     grid_locator1 = grid_finder.grid_locator1
        # if grid_locator2 is None:
        #     grid_locator2 = grid_finder.grid_locator2
        # if tick_formatter1 is None:
        #     tick_formatter1 = grid_finder.tick_formatter1
        # if tick_formatter2 is None:
        #     tick_formatter2 = grid_finder.tick_formatter2

        aux_trans = self.grid_finder.get_transform()
        return type(self)(aux_trans,
                          extreme_finder=extreme_finder,
                          grid_locator1=grid_locator1,
                          grid_locator2=grid_locator2,
                          tick_formatter1=tick_formatter1,
                          tick_formatter2=tick_formatter2)

    def new_grid_helper_for_floating_axes(self):
        pass

    # def update_lim(self, axes):
    #     print("555")
    #     return super().update_lim(axes)

    def invalidate(self):
        # update_lim method internally check if the xlim and ylim and skips if
        # there have not changed. You should call invalidate method to force
        # the update. This is required, for example, if you have changed the
        # locator_params.
        self._old_limits = None


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
                          extreme_mode="host",
                          ticklabel_angle_mode="axis",
                          ):
        if axes is None:
            if isinstance(self, ParasiteAxesBase):
                axes = self._parent_axes
            else:
                axes = self
        gh = self.get_grid_helper(proj_name)
        axis = gh.new_floating_axis(nth_coord, value,
                                    axis_direction=axis_direction,
                                    axes=axes,
                                    extreme_mode="host",
                                    ticklabel_angle_mode=ticklabel_angle_mode,
                                    )
        return axis

    # This was deprecated in mpl3.6 and removed in mpl3.7. in the matplotlib,
    # we just revive it.
    def new_gridlines(self, grid_helper=None):
        """
        Create and return a new GridlineCollection instance.

        *which* : "major" or "minor"
        *axis* : "both", "x" or "y"

        """
        if grid_helper is None:
            grid_helper = self.get_grid_helper()

        gridlines = grid_helper.new_gridlines(self)
        return gridlines


    def locator_params(self, axis='both', **kwargs):

        gh = self.get_grid_helper()
        if isinstance(gh, GridHelperRectlinear):
            super().locator_params(axis=axis, **kwargs)
        else:
            _api.check_in_list(['x', 'y', 'both'], axis=axis)
            update_x = axis in ['x', 'both']
            update_y = axis in ['y', 'both']
            if update_x:
                l = gh.grid_finder.grid_locator1
                l.set_params(**kwargs)
            if update_y:
                l = gh.grid_finder.grid_locator2
                l.set_params(**kwargs)

            gh.invalidate()

        self.stale = True

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

    def get_floating_axes(self, proj_name, extremes, axes_class=None):
        # FIXME We should reorganize the class hierachy so that this method is
        # no availble to parasite axes.
        if isinstance(self, ParasiteAxesBase):
            raise RuntimeError("do not call this method from the parasite axes")

        if axes_class is None:
            axes_class = FloatingParasiteAxes # ParasiteAxes

        aux_trans = self.get_axisartist_transform(proj_name)
        # ax = self._add_twin_axes(
        #     axes_class, aux_transform=aux_trans, viewlim_mode="transform")
        grid_helper = self.get_grid_helper(proj_name).copy()

        ax = self._add_twin_axes(
            axes_class, aux_transform=aux_trans, viewlim_mode=None,
            boundary=extremes, grid_helper=grid_helper)

        # FIXME This is very hacky, but we reassign axis and gridlines. We
        # should have a better way of doing this. Parasite axes should have an
        # attribute to its host axes, and the host axes should be used to
        # created axis and gridlines.
        # ax._grid_helper = grid_helper

        new_floating_axis = grid_helper.new_floating_axis
        for loc, nth_coord, v in zip(["left", "right", "bottom", "top", ],
                                     [0, 0, 1, 1], # nth_coord
                                     extremes):
            # ax._axislines[loc] = new_floating_axis(loc=loc, axes=self,
            #                                        axis_direction=loc)
            ax.axis[loc] = new_floating_axis(nth_coord, v,
                                             axis_direction=loc,
                                             axes=self,
                                             extreme_mode="self")

        ax.axis["bottom"].get_helper().set_extremes(*extremes[:2])
        ax.axis["top"].get_helper().set_extremes(*extremes[:2])
        ax.axis["left"].get_helper().set_extremes(*extremes[2:])
        ax.axis["right"].get_helper().set_extremes(*extremes[2:])

        ax.axis["top", "right"].toggle(ticklabels=False)

        ax.locator_params(nbins=5) # default value is usually too dense.

        return ax

from mpl_toolkits.axes_grid1.parasite_axes import (
    ParasiteAxesBase, host_axes_class_factory, parasite_axes_class_factory)
#     parasite_axes_auxtrans_class_factory, subplot_class_factory)


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
import matplotlib.patches as mpatches

class FloatingParasiteAxes(ParasiteAxes):

    def __init__(self, *args, **kwargs):
        self._boundary = kwargs.pop("boundary")
        grid_helper = kwargs.get("grid_helper", None)
        if grid_helper is None:
            raise ValueError("FloatingAxes requires grid_helper argument")
        # if not hasattr(grid_helper, "get_boundary"):
        #     raise ValueError("grid_helper must implement get_boundary method")

        # We need to force vewlim_mode of None, so that we can use set_xlim and
        # set_ylim methods.
        # kwargs["viewlim_mode"] = None
        kwargs["autoscale_on"] = False
        super().__init__(*args, **kwargs)
        # self.set_aspect(1.)

    def _gen_axes_patch(self):
        # docstring inherited
        # Using a public API to access _extremes.
        # (x0, _), (x1, _), (y0, _), (y1, _) = map(
        #     self.get_grid_helper().get_data_boundary,
        #     ["left", "right", "bottom", "top"])
        x0, x1, y0, y1 = self._boundary
        patch = mpatches.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        patch.get_path()._interpolation_steps = 100
        return patch

    def _update_axes_patch(self):
        x0, x1, y0, y1 = self._boundary
        self.patch.set_xy([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    def cla(self):
        super().cla()
        self.patch.set_transform(
            # self.get_grid_helper().grid_finder.get_transform()
            self.transData)
        # The original patch is not in the draw tree; it is only used for
        # clipping purposes.
        orig_patch = super()._gen_axes_patch()
        orig_patch.set_figure(self.figure)
        orig_patch.set_transform(self.transAxes)
        self.patch.set_clip_path(orig_patch)
        # self.gridlines.set_clip_path(orig_patch)

    def set_xlim(self, left=None, right=None, emit=True, auto=False,
                 *, xmin=None, xmax=None):
        print("viewmode", )
        if right is None and np.iterable(left):
            left, right = left
        if xmin is not None:
            if left is not None:
                raise TypeError("Cannot pass both 'left' and 'xmin'")
            left = xmin
        if xmax is not None:
            if right is not None:
                raise TypeError("Cannot pass both 'right' and 'xmax'")
            right = xmax

        boundary = (left, right) + tuple(self._boundary[2:])
        self.update_boundary(*boundary)

    def set_ylim(self, bottom=None, top=None, emit=True, auto=False,
                 *, ymin=None, ymax=None):
        if top is None and np.iterable(bottom):
            bottom, top = bottom
        if ymin is not None:
            if bottom is not None:
                raise TypeError("Cannot pass both 'bottom' and 'ymin'")
            bottom = ymin
        if ymax is not None:
            if top is not None:
                raise TypeError("Cannot pass both 'top' and 'ymax'")
            top = ymax

        boundary = tuple(self._boundary[:2]) + (bottom, top)
        self.update_boundary(*boundary)

    def update_boundary(self, left, right, bottom, top):
        # Since it parasite axes with viewLim synced to the host, changing the
        # boundary by overriding set_xlim and set_ylim does not work well.

        # we update self.patch so that we have a correct clipping path.
        self._boundary[:] = left, right, bottom, top
        self._update_axes_patch()

        # update xlim
        for loc in ["bottom", "top"]:
            self.axis[loc].get_helper().set_extremes(left, right)
        for loc, v in zip(["left", "right"], [left, right]):
            self.axis[loc].get_helper().value = v

        # update ylim
        for loc in ["left", "right"]:
            self.axis[loc].get_helper().set_extremes(bottom, top)
        for loc, v in zip(["bottom", "top"], [bottom, top]):
            self.axis[loc].get_helper().value = v

    def adjust_host_axes_lim(self, host_ax):
        bbox = self.patch.get_path().get_extents(
            # First transform to pixel coords, then to parent data coords.
            self.patch.get_transform() - host_ax.transData)
        bbox = bbox.expanded(1.02, 1.02)
        host_ax.set_xlim(bbox.xmin, bbox.xmax)
        host_ax.set_ylim(bbox.ymin, bbox.ymax)

    def add_artist(self, a):
        super().add_artist(a)
        print("add")
        a.set_clip_box(self._parent_axes.bbox)

def test1():
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

    # We create new parasite axes to draw a x-axis of the rotated axes. Note
    # that the transData of this axes is still that of the original axes, and
    # we cannot use it to draw histogram.


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

    ax.locator_params(steps=[1., 5, 10], nbins=8)
    ax_rotated.locator_params(steps=[1., 5, 10], nbins=3)

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
                                                          "top")
    # ax_rotated.axis["y=2"].set_transform(ax.transData)

    # FIXME maybe we introduce axis.set_lim method?
    # ax_rotated.axis["y=2"].get_helper().set_extremes(0.5, 2) # limit its extents.
    ax_rotated.axis["y=2"].set_lim(0.5, 2) # limit its extents.

    ax_rotated.axis["x=2"] = ax_rotated.new_floating_axis(0, 2,
                                                          "right")
    # ax_rotated.axis["x=2"].get_helper().set_extremes(1, 2.5) # limit its extents.
    ax_rotated.axis["x=2"].set_lim(1, 2.5) # limit its extents.
    ax_rotated.axis["x=2"].set_axis_direction("right")

    # ax.new_floating_axis(proj_name, nth_coord=0, trail=trail, dir="left")
    # extent = [0, 0, 1, 1]
    # ax_floating = ax.new_floating_axes(proj_name, extent)

def test2():
    import matplotlib.pyplot as plt
    plt.clf()
    ax = plt.subplot(111, axes_class=HostAxes, aspect=1)

    from matplotlib.transforms import Affine2D
    # We define transform from the original axes to the rotated one.
    transform_to_rotated = (
        Affine2D()
        .translate(0, 0)
        .rotate_deg(15)
        .translate(0, 0)
        .scale(1, 1) # to reduce the height of the histogram
    )

    transform_from_rotated = transform_to_rotated.inverted()

    proj_name = "rotated"
    ax.register_projection(proj_name,
                           transform_from_rotated, grid_helper=None)

    # ax.add_twin(name="rotaed", projection=None)

    # ax.select_visible_projection(proj_name) # to replace ticks and ticklabels.
    # ax.grid(True)

    ax_rotated = ax.get_projected_axes(proj_name)
    ax_rotated.locator_params(steps=[1, 2, 5, 10], nbins=10)

    ax_rotated.grid(color="r", alpha=0.2, zorder=0)


    ax.plot([0], [0], "bo", ms=10, mfc="none")
    ax_rotated.plot([0, 0], [0, 1], "ro")

    extremes = [0, 1.5, 1., 2.5]
    ax_floating = ax.get_floating_axes(proj_name, extremes)

    # ax_floating.locator_params(steps=[1., 5, 10], nbins=100)


    # gh = ax_rotated.get_grid_helper()
    # l = gh.grid_finder.grid_locator1
    # l.set_params(steps=[1., 5,10])
    # l.set_params(nbins=10)
    # l = gh.grid_finder.grid_locator2
    # l.set_params(steps=[1., 5, 10])
    # l.set_params(nbins=10)

    # Are tick locations cached for same axes data limits? If xlim ylim is
    # changed early in the code, the above set_params is reflected only for the
    # floating axes, not for the rotated axes.
    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-0.5, 2.5)

    ax.set_aspect(1)

    ax_floating.plot([-0.5, 2], [0., 2.5])

    # for the floating axes, it would be better to have a separate grid_helper
    # instance so that the params does not affect other axes. For now, this
    # will override previous locator_params on the ax_rotated.
    ax_floating.locator_params(nbins=5)
    # ax_floating.update_boundary(0, 2, 1, 2)
    ax_floating.set_xlim(1, 2)
    ax_floating.set_ylim(1, 2.5)
    # ax_floating.set_ylim(1, 2.5)
    # ax_floating.set_xlim(0, 1.5)
    ax_floating.patch.set_visible(True)

    # ax_floating.plot([0., 2.5], [0., 2.5], clip_on=False)

    ax_floating.axis["bottom"].label.set_text("Test")

def test3():
    import matplotlib.pyplot as plt
    plt.clf()
    ax = plt.subplot(111, axes_class=HostAxes, aspect=1)
    # ax.set_aspect(1)

    from matplotlib.transforms import Affine2D
    # We define transform from the original axes to the rotated one.
    transform_to_rotated = (Affine2D()
                 .skew_deg(0, -30)
                 .translate(0, 0)
                 .scale(1, 1) # to reduce the height of the histogram
    )

    transform_from_rotated = transform_to_rotated.inverted()

    proj_name = "rotated"
    ax.register_projection(proj_name,
                           transform_from_rotated, grid_helper=None)

    # ax.add_twin(name="rotaed", projection=None)

    # ax.select_visible_projection(proj_name) # to replace ticks and ticklabels.
    # ax.grid(True)

    ax_rotated = ax.get_projected_axes(proj_name)

    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-0.5, 2.5)

    ax_rotated.locator_params(steps=[1, 2, 5, 10], nbins=10)

    ax_rotated.grid(color="r", alpha=0.2, zorder=0)


    ax.plot([0], [0], "bo", ms=10, mfc="none")
    ax_rotated.plot([0, 0], [0, 1], "ro")

    extremes = [0, 1.5, 1., 2.5]
    ax_floating = ax.get_floating_axes(proj_name, extremes)

    # ax_floating.locator_params(steps=[1., 5, 10], nbins=100)

    # for the floating axes, it would be better to have a separate grid_helper
    # instance so that the params does not affect other axes. For now, this
    # will override previous locator_params on the ax_rotated.
    ax_floating.locator_params(nbins=5)
    ax_floating.update_boundary(0, 2, 0, 2)
    # ax_floating.patch.set_visible(True)

    # ax_floating.plot([0., 2.5], [0., 2.5], clip_on=False)


    f1 = ax_floating.fill_between([0., 2], [0., 4.], alpha=0.2)

    # FIXME This should be automatically done.
    # f1.set_clip_box(ax.bbox)

    im = ax_floating.imshow(np.arange(10).reshape((10, 1)), extent=[0, 2, 0, 2],
                            origin="lower", interpolation="none")
    # im.remove()

    # from mpl_visual_context.image_box_effect import BboxImageEffect
    # im.axes = ax_floating
    # f1.set_path_effects([BboxImageEffect(im)])

def test4():
# if True:
    import matplotlib.pyplot as plt
    plt.clf()
    ax = plt.subplot(111, axes_class=HostAxes, aspect="auto")

    from matplotlib.transforms import Affine2D
    # We define transform from the original axes to the rotated one.
    transform_to_rotated = (Affine2D()
                 .skew_deg(0, -30)
                 .translate(0, 0)
                 .scale(1, 1) # to reduce the height of the histogram
    )

    transform_from_rotated = transform_to_rotated.inverted()

    proj_name = "rotated"
    ax.register_projection(proj_name,
                           transform_from_rotated, grid_helper=None)

    # ax.add_twin(name="rotaed", projection=None)

    # ax.select_visible_projection(proj_name) # to replace ticks and ticklabels.
    # ax.grid(True)

    ax_rotated = ax.get_projected_axes(proj_name)
    ax_rotated.grid(color="r", alpha=0.2, zorder=0)


    f1 = ax_rotated.fill_between([0., 1, 2], [0., 3, 2.])
    # im = ax_rotated.imshow(np.arange(10).reshape((10, 1)), extent=[0, 2, 0, 3],
    #                        origin="lower", interpolation="bilinear")
    #  f1.set_path_effects([ImageClipEffect(im, remove_from_axes=True)])

    # im.remove()
    # im.axes = ax_rotated

    # from mpl_visual_context.image_box_effect import GradientEffect
    from mpl_visual_context.patheffects import AlphaGradient

    f1.set_path_effects([AlphaGradient("up")])

    ax.set(xlim=(0, 3), ylim=(0, 4))

    plt.show()

def test5():
    from matplotlib.transforms import Affine2D
    # import mpl_toolkits.axisartist.floating_axes as floating_axes
    import numpy as np
    from mpl_toolkits.axisartist.grid_finder import MaxNLocator
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4, 4), num=2)
    fig.clf()
    ax = fig.add_subplot(111, axes_class=HostAxes, aspect=1)

    tr_translate = Affine2D().translate(0, 0)
    tr_scale = Affine2D().scale(1, 1)

    # tr_ = Affine2D().rotate_deg(-45) + tr_scale + tr_translate 
    # tr = Affine2D().translate(-x1, -y1).scale(1/dx, 1/dy).rotate_deg(45)
    tr = tr_translate + tr_scale + Affine2D().rotate_deg(45)

    proj_name = "rotated"
    ax.register_projection(proj_name,
                           tr, grid_helper=None)

    # ax_rotated = ax.get_projected_axes(proj_name)
    # ax_rotated.grid(color="r", alpha=0.2, zorder=0)



    extremes = [0, 1, 0, 1]
    rot_ax = ax.get_floating_axes(proj_name, extremes)

    def update_lim(x1, x2, y1, y2):
        dx = x2 - x1
        dy = y2 - y1

        tr_translate.clear().translate(-x1, -y1)
        tr_scale.clear().scale(1/dx, 1/dy)

        rot_ax.update_boundary(x1, x2, y1, y2)
        # tr = tr_.inverted()

        fig.canvas.draw_idle()

    # ax.set_xlim(-0.7, 0.7)
    # ax.set_ylim(0., 1.4)

    rot_ax.axis['bottom'].major_ticklabels.set(rotation=-45, ha='left', va='top')
    rot_ax.axis['bottom'].label.set(text="Variable A",
                                    rotation=-45, ha='left', va='top')
    rot_ax.axis['left'].major_ticklabels.set(rotation=45, ha='right', va='top')
    rot_ax.axis['left'].label.set(text="Variable B",
                                  rotation=45, ha='right', va='top')

    ax.axis[:].set_visible(False)
    ax.patch.set_visible(False)

    rot_ax.adjust_host_axes_lim(ax)

    x1, x2 = 3, 4
    y1, y2 = 1, 5
    update_lim(x1, x2, y1, y2)

    rot_ax.scatter(np.random.rand(10, 1)*(x2-x1)+x1,
                   np.random.rand(10, 1)*(y2-y1)+y1)
