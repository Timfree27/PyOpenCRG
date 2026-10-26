# OpenCRG MATLAB tools -> pyopencrg

`road` is a `pyopencrg.CrgRoad`, `crg` is `import pyopencrg as crg`.

## Reading, writing, checking

| MATLAB | pyopencrg |
|---|---|
| `crg_read(file)` | `crg.read(file)` |
| `crg_write(data, file, type)` | `road.save(file, fmt)` |
| `crg_demo_convert2bin` / `crg_demo_convert2txt` | `crg.convert(src, dst, "KRBI" / "LDFI")` |
| `ipl_read`, `ipl_write` | `crg.io.read_ipl`, `crg.io.write_ipl` |
| `sdf_cut`, `sdf_add` | `crg.io.cut_block` (adding is done by the writer) |
| `str_num2strn` | `crg.io.format_values` (vectorised) |
| `crg_check`, `crg_check_head`, `crg_check_data`, `crg_check_opts`, `crg_check_mods`, `crg_check_mpro`, `crg_check_wgs84`, `crg_check_curvature`, `crg_check_single` | `road.check()` |
| `crg_check_uv_descript` | validation inside `road.add_profile` / `road.add_random` |
| `crg_single` | `road.as_single()` (data is float64 otherwise) |
| `crg_wrap` | automatic (headings are always computed in [-pi, pi]) |
| `crg_mods` | `road.apply_mods()` |
| `crg_isequal` | `crg.compare(a, b)` |
| `crg_show_isequal` | `crg.compare(a, b).show()` |

## Evaluation

| MATLAB | pyopencrg |
|---|---|
| `crg_eval_uv2z` | `road.uv_to_z(u, v)` |
| `crg_eval_uv2xy` | `road.uv_to_xy(u, v)` |
| `crg_eval_xy2uv` | `road.xy_to_uv(x, y, u_hint=None)` |
| `crg_eval_xy2z` | `road.xy_to_z(x, y)` |
| `crg_eval_u2phi` | `road.u_to_phi(u)` |
| `crg_eval_u2crv` | `road.u_to_curvature(u)` |
| `crg_eval_uv2iuiv` | `road.uv_to_index(u, v)` (0 based) |
| `crg_eval_xyz2enh`, `crg_eval_enh2xyz` | `road.xyz_to_enh`, `road.enh_to_xyz` |
| `crg_eval_xyz2llh`, `crg_eval_llh2xyz` | `road.xyz_to_llh`, `road.llh_to_xyz` (degrees) |

## Editing

| MATLAB | pyopencrg |
|---|---|
| `crg_cut_iuiv` | `road.crop_index(iu, iv)` or `road.crop(u=..., v=...)` in metres |
| `crg_append` | `road.append(other)`, `crg.concatenate([...])` |
| `crg_flip` | `road.flip()` |
| `crg_filter` | `road.filter(method, size, weight, repeat, iu, iv)` |
| `crg_limiter` | `road.clip_z(z_min, z_max, iu, iv)` |
| `crg_peakfinder` | `road.find_peaks(threshold, size, iu, iv)` |
| `crg_rerender` | `road.resample(u_inc, v_inc, v)` |
| `crg_map_uv2uv` | `road.overlay_uv(other, iu, iv)` |
| `crg_map_xy2xy` | `road.overlay_xy(other, iu, iv)` |
| `crg_b2z` | `road.set_banking(b)` |
| `crg_s2z` | `road.set_reference_elevation(zr)`, `road.set_slope(s)` |
| `crg_ext_slope` | `road.extract_slope(p)` |
| `crg_ext_banking` | `road.extract_banking(p)` |
| `crg_separate_sb` | `road.separate_slope_banking(swlen, bwlen)` |
| `crg_perform2surface` | `road.add_profile(...)`, `road.add_random(...)` |
| `smooth_firfilt` | `crg.smoothing.fir_smooth` |
| `spl_smooth` (inside `crg_ext_*`) | `crg.smoothing.smoothing_spline` |

## Generation

| MATLAB | pyopencrg |
|---|---|
| `crg_gen_csb2crg0` | `crg.generate.synthetic_road(length, width, u_inc, v_inc, curvature, slope, banking)` |
| `crg_gen_pxy2ppxy` + `crg_gen_ppxy2phi` | `crg.generate.reference_line_from_points(x, y, u_inc)` / `road_from_points` |
| minimal struct (`data.u`, `data.v`, `data.z`) | `crg.CrgRoad(z, u_inc, v)` |
| `crg_demo_gen_*` | `examples/` folder |

## WGS84 and map projections

| MATLAB | pyopencrg |
|---|---|
| `crg_wgs84_dist` | `crg.wgs84.distance(lat1, lon1, lat2, lon2)` |
| `crg_wgs84_invdist` | `crg.wgs84.destination(lat, lon, azimuth, dist)` |
| `crg_wgs84_xy2wgs` | `road.xy_to_wgs84(x, y)` |
| `crg_wgs84_wgsxy2wgs` | `crg.wgs84.local_xy_to_wgs84(...)` |
| `crg_wgs84_setend` | `road.set_wgs84_end(azimuth)` |
| `crg_wgs84_wgs2url` | `crg.wgs84.to_url(lat, lon)` (OpenStreetMap or Google) |
| `crg_wgs84_crg2html`, `map_wgs2html` | `road.to_html_map(path)` (Leaflet + OpenStreetMap) |
| `map_check*`, `map_geod2ecef`, `map_ecef2geod`, `map_ecef2ecef` | `crg.mapping.MapProjection`, `geod_to_ecef`, `ecef_to_geod`, `helmert` |
| `map_geod2pmap(_tm)`, `map_pmap2geod_tm`, `map_ptm_*` | `crg.mapping.tm_forward`, `tm_inverse` |
| `map_global2plocal`, `map_plocal2global` | `MapProjection.to_local`, `MapProjection.to_global` |

## Plots

| MATLAB | pyopencrg |
|---|---|
| `crg_show` | `road.show()` |
| `crg_show_refline_map` | `road.plot.show_refline_map()` |
| `crg_show_refline_elevation` | `road.plot.show_refline_elevation()` |
| `crg_show_elgrid_cuts_and_limits` | `road.plot.show_cuts_and_limits()` |
| `crg_show_elgrid_surface` | `road.plot.show_elgrid_surface()` |
| `crg_show_road_surface` | `road.plot.show_road_surface()` |
| `crg_show_road_uv2surface` | `road.plot.show_road_uv_surface(u, v)` |
| `crg_show_info` | `print(road)`, `road.plot.show_info()` |
| `crg_show_peaks` | `road.plot.peaks(road.find_peaks())` |
| `crg_show_refpnts_and_refline`, `crg_plot_refpnt_distances` | `road.plot.show_refpoints(x, y)`, `road.plot.refpoint_distances(x, y)` |
| `crg_plot_refline_xy_overview_map` | `road.plot.refline_xy()` |
| `crg_plot_refline_heading` | `road.plot.heading()` |
| `crg_plot_refline_xy_map_and_curv` | `road.plot.refline_xy_curvature()` |
| `crg_plot_refline_curvature` | `road.plot.curvature()` |
| `crg_plot_refline_elevation` | `road.plot.refline_elevation()` |
| `crg_plot_refline_slope_bank` | `road.plot.slope_banking()` |
| `crg_plot_refline_xyz_map` | `road.plot.refline_xyz()` |
| `crg_plot_elgrid_limits` | `road.plot.grid_limits()` |
| `crg_plot_elgrid_cross_sect`, `crg_plot_elgrid_long_sect` | `road.plot.cross_sections()`, `road.plot.long_sections()` |
| `crg_plot_elgrid_uvz_map`, `crg_plot_elgrid_xyz_map` | `road.plot.elgrid_uvz_map()`, `road.plot.elgrid_xyz_map()` |
| `crg_plot_road_uvz_map`, `crg_plot_road_xyz_map` | `road.plot.road_uvz_map()`, `road.plot.road_xyz_map()` |
| `crg_plot_road_uv2uvz_map`, `crg_plot_road_uv2xyz_map` | `road.plot.show_road_uv_surface(u, v)` |
| `crg_figure`, `copy_ax2fig`, `crg_surf` | internal helpers; every plot accepts `ax=` |

## Differences worth knowing

* Data is kept in **float64** (MATLAB uses single). KRBI files are still written in float32.
* Indices are **0 based** and both ends are included in `crop_index`.
* Editing methods return a new road (MATLAB style), the input is never modified.
* NaN values *inside* a cross section are accepted (MATLAB refuses them); `check()` reports them.
* `filter()` filters only the elevation grid; slope and banking are kept (MATLAB deletes them).
* `clip_z()` limits the absolute elevation and keeps slope and banking.
* `synthetic_road()` uses the heading at the middle of each interval, so that the reference line
  of a circle is exactly on the circle.
* `read()` checks the file and refuses truncated or invalid files (MATLAB only warns about
  truncated data). `check=False` opens them anyway.
* `crg_check_data` rounds u and v values to multiples of 0.001 m; pyopencrg keeps them exact and
  only warns.
* `xy_to_uv` starts its search from the closest reference line point (KD-tree) instead of the
  evaluation history (use `u_hint` for roads that cross themselves).
* Only the options that change a result are available in Python (border modes, smoothing zones,
  track closing, check tolerances). The options of other tools (`refline_search_*`, `warn_*`,
  `log_*`) are not exposed, but they are kept unchanged when a file is read and saved again.
* The curvature checks always run (MATLAB lets `warn_curv_local/global` switch them).
* `fill_nan()` / `grid_nan_mode = 2` also interpolates NaN values *inside* a cross section
  (MATLAB refuses such files).
* `resample()` samples the reference line with a cubic spline and re-integrates the headings
  (MATLAB walks along the spline); both differ by less than 1 mm.
* `add_random()` uses the numpy random generator: the same seed does not give the same surface
  as MATLAB. With `operation="multiply"` only the area covered by the sections is multiplied.
* `overlay_uv` / `overlay_xy` add the other road to the grid and keep slope and banking.
* Figure options of `data.fopt` (colormap modulo, light, aspect) are not supported; every plot
  accepts an `ax` argument instead.
