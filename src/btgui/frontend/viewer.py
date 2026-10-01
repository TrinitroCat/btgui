"""PyVista-backed 3D structure rendering and direct mouse interaction."""

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QRubberBand
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkRenderingCore import vtkLight

from btgui.backend.core import angle_between, distance_between
from btgui.data.appearance import covalent_radius, default_color, vdw_radius


class StructureViewer(QtInteractor):
    """Render raw structure lists with VTK world-coordinate geometry.

    Args:
        parent: Optional Qt parent widget.

    Return:
        A Qt widget containing the interactive PyVista renderer.
    """

    coordinatesEdited = Signal(int, object, object)
    deleteRequested = Signal(int, object)
    atomsAdded = Signal(int, object, object)
    filesDropped = Signal(object)
    measurementModeChanged = Signal(object)
    selectionChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.elements = []
        self.coordinates = []
        self.lattices = []
        self.current_index = 0
        self.selected_indices = np.empty(0, dtype=int)
        self.property_renderers = {}
        self.atom_styles = {}
        self.atom_frame_styles = []
        self.frame_metrics = []
        self.default_atom_style = "covalent"
        self.default_custom_radius = 0.76
        self._background_color_value = "#f5f7f7"
        self.bonds_enabled = False
        self.bond_tolerance = 1.2
        self.bond_width = 0.08
        self.bond_alpha = 1.0
        self.axes_visible = False
        self.grid_visible = False
        self.atom_material = "default"
        self.material_ambient = 0.15
        self.material_diffuse = 0.78
        self.material_specular = 0.25
        self.material_roughness = 0.42
        self.material_metallic = 0.0
        self.atom_lighting = True
        self.atom_edges = False
        self.atom_edge_color = "#202020"
        self.atom_edge_width = 1.0
        self.error_handler = None
        self.pressed_button = Qt.MouseButton.NoButton
        self.press_position = None
        self.press_modifiers = Qt.KeyboardModifier.NoModifier
        self.drag_mode = None
        self.dragged = False
        self.drag_start_coordinates = None
        self.drag_start_positions = None
        self.drag_start_mouse = None
        self.drag_start_basis = None
        self.drag_start_depth = None
        self.drag_start_world = None
        self.drag_center = None
        self.drag_center_screen = None
        self.drag_rotation_zone = None
        self.drag_last_mouse = None
        self.drag_outer_angle = 0.0
        self.drag_preview_matrix = np.eye(4)
        self.selected_actors = []
        self.selection_shell_actors = []
        self.bond_actors = []
        self.atom_meshes = []
        self.lattice_actors = []
        self.measurement_actors = []
        self.playback_active = False
        self._playback_base_positions = None
        self._playback_base_elements = None
        self.zoom_percent = 100
        self._zoom_base_distance = None
        self.length_measurements = {}
        self.angle_measurements = {}
        self.measurement_mode = None
        self.measurement_selection = []
        self.selection_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.selection_band.hide()
        self.selection_start = None
        self.add_mode = False
        self.add_symbol = None
        self.add_reference_position = None
        self.add_pending_elements = []
        self.add_pending_positions = []
        self.add_active_actor = None
        self.add_fixed_actors = []
        self._add_preview_position = None
        self.add_reference_symbol = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.set_background(self._background_color_value)
        self._configure_lights()
        self.enable_anti_aliasing("msaa")

    def _configure_lights(self):
        """Install a key and fill light for shaded sphere rendering.

        Args:
            None.

        Return:
            None. Scene lights are retained by the VTK renderer.
        """
        lights = self.renderer.GetLights()
        if lights is not None and lights.GetNumberOfItems() > 0:
            self.renderer.RemoveAllLights()
        key_light = vtkLight()
        key_light.SetLightTypeToSceneLight()
        key_light.SetPosition(1.0, -1.0, 1.5)
        key_light.SetFocalPoint(0.0, 0.0, 0.0)
        key_light.SetIntensity(1.0)
        fill_light = vtkLight()
        fill_light.SetLightTypeToSceneLight()
        fill_light.SetPosition(-1.0, 0.5, 0.5)
        fill_light.SetFocalPoint(0.0, 0.0, 0.0)
        fill_light.SetIntensity(0.35)
        self.renderer.AddLight(key_light)
        self.renderer.AddLight(fill_light)

    def set_data(self, elements, coordinates, lattices):
        """Attach the caller's three frame-ordered lists without wrapping them.

        Args:
            elements: Per-frame element arrays.
            coordinates: Per-frame Cartesian position arrays.
            lattices: Per-frame lattice matrices.

        Return:
            None. The first frame is displayed with a fitted camera.
        """
        if self.add_mode:
            self.cancel_atom_addition()
        self.elements = elements
        self.coordinates = coordinates
        self.lattices = lattices
        self.current_index = 0
        self.selected_indices = np.empty(0, dtype=int)
        self.length_measurements = {}
        self.angle_measurements = {}
        self.set_measurement_mode(None)
        self.draw_frame(reset_camera=True)

    def set_frame_state(self, atom_styles, metrics):
        """Attach frame-aligned atom styles and measurement mappings.

        Args:
            atom_styles: Per-frame lists of per-atom style dictionaries.
            metrics: Per-frame dictionaries containing length and angle mappings.

        Return:
            None. The provided containers remain owned by the main window.
        """
        self.atom_frame_styles = atom_styles
        self.frame_metrics = metrics
        if 0 <= self.current_index < len(metrics):
            self.length_measurements = metrics[self.current_index]["length"]
            self.angle_measurements = metrics[self.current_index]["angle"]
        else:
            self.length_measurements = {}
            self.angle_measurements = {}

    def set_frame(self, index):
        """Display one frame while retaining the current camera.

        Args:
            index: Frame index to display.

        Return:
            None. Selection is cleared for the new frame.
        """
        if not 0 <= index < len(self.coordinates):
            raise IndexError("frame index is out of range")
        self.current_index = index
        self.selected_indices = np.empty(0, dtype=int)
        if 0 <= index < len(self.frame_metrics):
            self.length_measurements = self.frame_metrics[index]["length"]
            self.angle_measurements = self.frame_metrics[index]["angle"]
        else:
            self.length_measurements = {}
            self.angle_measurements = {}
        self.draw_frame()

    def draw_frame(self, reset_camera=False):
        """Rebuild the current VTK scene from the attached raw arrays.

        Args:
            reset_camera: Whether to fit and reset the camera after rebuilding.

        Return:
            None. Atom coordinates remain owned by the caller and are not copied.
        """
        camera_state = None if reset_camera else self._camera_state()
        self.clear()
        self._configure_lights()
        self.selected_actors = []
        self.selection_shell_actors = []
        self.bond_actors = []
        self.atom_meshes = []
        self.lattice_actors = []
        self.measurement_actors = []
        has_frame = 0 <= self.current_index < len(self.coordinates)
        if not has_frame:
            self.render()
            return

        positions = np.asarray(self.coordinates[self.current_index], dtype=float)
        elements = np.asarray(self.elements[self.current_index], dtype=str)

        # Section: atoms. Group identical styles so each group is one glyph mesh.
        groups = {}
        selected_mask = np.zeros(len(elements), dtype=bool)
        valid_selected = self.selected_indices[
            (self.selected_indices >= 0) & (self.selected_indices < len(elements))
        ]
        selected_mask[valid_selected] = True
        for index, symbol in enumerate(elements):
            style = self._style_for(symbol, index)
            radius = self._style_radius(symbol, index)
            key = (style["color"], style["alpha"], radius, selected_mask[index])
            groups.setdefault(key, []).append(index)
        material = self._material_properties()
        for (color, alpha, radius, is_selected), indices in groups.items():
            source = pv.Sphere(radius=radius, theta_resolution=24, phi_resolution=16)
            glyphs = pv.PolyData(positions[np.asarray(indices, dtype=int)]).glyph(
                orient=False,
                scale=False,
                geom=source,
            )
            actor = self.add_mesh(
                glyphs,
                color=color,
                opacity=alpha,
                smooth_shading=True,
                lighting=self.atom_lighting,
                show_edges=self.atom_edges,
                edge_color=self.atom_edge_color,
                line_width=self.atom_edge_width,
                ambient=material["ambient"],
                diffuse=material["diffuse"],
                specular=material["specular"],
                specular_power=material["specular_power"],
                render=False,
            )
            actor_property = actor.GetProperty()
            has_pbr = hasattr(actor_property, "SetInterpolationToPBR")
            if has_pbr:
                actor_property.SetInterpolationToPBR()
                if hasattr(actor_property, "SetMetallic"):
                    actor_property.SetMetallic(material["metallic"])
                if hasattr(actor_property, "SetRoughness"):
                    actor_property.SetRoughness(material["roughness"])
            else:
                actor_property.SetInterpolationToPhong()
            actor_property.SetAmbient(material["ambient"])
            actor_property.SetDiffuse(material["diffuse"])
            actor_property.SetSpecular(material["specular"])
            actor_property.SetSpecularPower(material["specular_power"])
            self.atom_meshes.append((glyphs, np.asarray(indices, dtype=int), glyphs.points.copy()))
            if is_selected:
                self.selected_actors.append(actor)

        # Section: selection outline. A wireframe shell stays in world units.
        if len(valid_selected):
            selected_elements = elements[valid_selected]
            for symbol in np.unique(selected_elements):
                indices = valid_selected[selected_elements == symbol]
                radius = max(
                    self._style_radius(elements[index], int(index)) for index in indices
                ) * 1.08 + 0.025
                source = pv.Sphere(radius=radius, theta_resolution=24, phi_resolution=16)
                shells = pv.PolyData(positions[indices]).glyph(
                    orient=False,
                    scale=False,
                    geom=source,
                )
                actor = self.add_mesh(
                    shells,
                    color="#18a999",
                    style="wireframe",
                    line_width=2.0,
                    lighting=False,
                    render=False,
                )
                self.selected_actors.append(actor)
                self.selection_shell_actors.append(actor)

        self._draw_lattice(np.asarray(self.lattices[self.current_index], dtype=float))
        if self.bonds_enabled:
            self._draw_bonds(positions, elements)
        self._draw_measurements(positions)
        for renderer in self.property_renderers.values():
            renderer(self, self.current_index, positions)

        self._apply_reference_widgets(positions, np.asarray(self.lattices[self.current_index], dtype=float))
        if reset_camera:
            self.camera_position = "iso"
            self.reset_camera()
            self._capture_zoom_reference()
        elif camera_state is not None:
            self._restore_camera_state(camera_state)
        self.render()

    def _camera_state(self):
        """Snapshot immutable numeric camera state before rebuilding actors.

        Args:
            None.

        Return:
            Numeric position, focal point, view-up, scale, angle, and clipping range.
        """
        return {
            "position": tuple(float(value) for value in self.camera.position),
            "focal_point": tuple(float(value) for value in self.camera.focal_point),
            "view_up": tuple(float(value) for value in self.camera.up),
            "parallel_scale": float(self.camera.parallel_scale),
            "view_angle": float(self.camera.view_angle),
            "clipping_range": tuple(float(value) for value in self.camera.clipping_range),
        }

    def _restore_camera_state(self, state):
        """Restore a numeric camera snapshot exactly after a scene rebuild.

        Args:
            state: Dictionary returned by :meth:`_camera_state`.

        Return:
            None. No zoom reference is recaptured.
        """
        self.camera.position = state["position"]
        self.camera.focal_point = state["focal_point"]
        self.camera.up = state["view_up"]
        self.camera.parallel_scale = state["parallel_scale"]
        self.camera.view_angle = state["view_angle"]
        self.camera.clipping_range = state["clipping_range"]

    def _style_for(self, symbol, atom_index=None):
        """Return one element style merged with defaults.

        Args:
            symbol: Chemical element symbol.

        Return:
            A dictionary containing display type, color, opacity, and radius.
        """
        style = self.atom_styles.get(symbol, {})
        if (
            atom_index is not None
            and 0 <= self.current_index < len(self.atom_frame_styles)
            and 0 <= atom_index < len(self.atom_frame_styles[self.current_index])
        ):
            style = {**style, **self.atom_frame_styles[self.current_index][atom_index]}
        return {
            "style": style.get("style", self.default_atom_style),
            "color": style.get("color", default_color(symbol)),
            "alpha": style.get("alpha", 1.0),
            "radius": style.get("radius", self.default_custom_radius),
        }

    def _material_properties(self):
        """Return VTK material coefficients for the selected preset.

        Args:
            None.

        Return:
            Material coefficients accepted by ``Plotter.add_mesh``.
        """
        presets = {
            "default": {"ambient": 0.15, "diffuse": 0.78, "specular": 0.25, "roughness": 0.42, "metallic": 0.0},
            "matte": {"ambient": 0.28, "diffuse": 0.72, "specular": 0.0, "roughness": 0.9, "metallic": 0.0},
            "glossy": {"ambient": 0.10, "diffuse": 0.72, "specular": 0.75, "roughness": 0.18, "metallic": 0.0},
            "metallic": {"ambient": 0.12, "diffuse": 0.55, "specular": 0.85, "roughness": 0.24, "metallic": 0.65},
        }
        material = dict(presets.get(self.atom_material, presets["default"]))
        material.update(
            ambient=self.material_ambient,
            diffuse=self.material_diffuse,
            specular=self.material_specular,
            roughness=self.material_roughness,
            metallic=self.material_metallic,
            specular_power=max(1.0, 128.0 * (1.0 - self.material_roughness)),
        )
        return material

    def _set_material_preset(self, material):
        """Load the coefficient defaults associated with one material preset."""
        presets = {
            "default": (0.24, 0.86, 0.32, 0.34, 0.0),
            "matte": (0.28, 0.72, 0.0, 0.9, 0.0),
            "glossy": (0.18, 0.86, 0.78, 0.16, 0.0),
            "metallic": (0.20, 0.76, 0.88, 0.22, 0.40),
        }
        self.material_ambient, self.material_diffuse, self.material_specular, self.material_roughness, self.material_metallic = presets[material]

    def _style_radius(self, symbol, atom_index=None):
        """Resolve one atom's physical display radius in angstroms.

        Args:
            symbol: Chemical element symbol.

        Return:
            Radius used by the VTK sphere geometry.
        """
        style = self._style_for(symbol, atom_index)
        if style["style"] == "point":
            return 0.12
        if style["style"] == "covalent":
            return covalent_radius(symbol)
        if style["style"] == "vdw":
            return vdw_radius(symbol)
        return max(0.01, float(style["radius"]))

    def _draw_bonds(self, positions, elements):
        """Draw Euclidean covalent-radius bonds as world-coordinate cylinders.

        Args:
            positions: Current frame Cartesian positions.
            elements: Current frame element symbols.

        Return:
            None. Bond actors are added to the current renderer.
        """
        radii = np.asarray([covalent_radius(symbol) for symbol in elements])
        meshes_by_color = {}
        for left in range(len(positions)):
            for right in range(left + 1, len(positions)):
                vector = positions[right] - positions[left]
                distance = float(np.linalg.norm(vector))
                if distance == 0 or distance > (radii[left] + radii[right]) * self.bond_tolerance:
                    continue
                color = self._blend_colors(
                    self._style_for(elements[left], left)["color"],
                    self._style_for(elements[right], right)["color"],
                )
                cylinder = pv.Cylinder(
                    center=(positions[left] + positions[right]) / 2,
                    direction=vector,
                    radius=self.bond_width,
                    height=distance,
                    resolution=16,
                    capping=True,
                )
                meshes_by_color.setdefault(color, []).append(cylinder)
        for color, meshes in meshes_by_color.items():
            mesh = meshes[0] if len(meshes) == 1 else pv.merge(meshes, merge_points=False)
            actor = self.add_mesh(
                mesh,
                color=color,
                opacity=self.bond_alpha,
                smooth_shading=True,
                render=False,
            )
            self.bond_actors.append(actor)

    def _blend_colors(self, first, second):
        """Return the visual midpoint of two hexadecimal colors.

        Args:
            first: First ``#rrggbb`` color.
            second: Second ``#rrggbb`` color.

        Return:
            A hexadecimal midpoint color.
        """
        values = [
            int(first[index:index + 2], 16) + int(second[index:index + 2], 16)
            for index in (1, 3, 5)
        ]
        return "#" + "".join(f"{value // 2:02x}" for value in values)

    def _draw_lattice(self, lattice):
        """Draw the current cell edges as VTK lines.

        Args:
            lattice: Row-vector lattice matrix.

        Return:
            None. One line mesh is added to the renderer.
        """
        corners = np.array(
            [
                [0.0, 0.0, 0.0],
                lattice[0], lattice[1], lattice[2],
                lattice[0] + lattice[1], lattice[0] + lattice[2],
                lattice[1] + lattice[2], lattice.sum(axis=0),
            ]
        )
        edges = (
            (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
            (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
        )
        cell = pv.PolyData(corners)
        cell.lines = np.asarray([[2, start, end] for start, end in edges], dtype=np.int64)
        self.lattice_actors.append(
            self.add_mesh(cell, color="#687577", line_width=1.5, lighting=False, render=False)
        )

    def _scene_bounds(self, positions, lattice):
        """Return padded world bounds containing atoms and the unit cell.

        Args:
            positions: Current frame Cartesian positions.
            lattice: Current frame lattice matrix.

        Return:
            VTK bounds ordered as xmin, xmax, ymin, ymax, zmin, zmax.
        """
        cell_points = np.array(
            [
                [0.0, 0.0, 0.0],
                lattice[0], lattice[1], lattice[2],
                lattice[0] + lattice[1], lattice[0] + lattice[2],
                lattice[1] + lattice[2], lattice.sum(axis=0),
            ]
        )
        points = np.concatenate((positions, cell_points), axis=0) if len(positions) else cell_points
        low = points.min(axis=0)
        high = points.max(axis=0)
        padding = np.maximum((high - low) * 0.05, 0.25)
        low -= padding
        high += padding
        return (low[0], high[0], low[1], high[1], low[2], high[2])

    def _apply_reference_widgets(self, positions, lattice):
        """Apply axes and fixed world-coordinate grid visibility.

        Args:
            positions: Current frame Cartesian positions.
            lattice: Current frame lattice matrix.

        Return:
            None. Renderer decorations are recreated after every scene clear.
        """
        if self.axes_visible:
            self.show_axes()
        else:
            self.hide_axes()
        if self.grid_visible:
            self.show_grid(
                bounds=self._scene_bounds(positions, lattice),
                color="#8a9597",
                font_size=10,
            )
        else:
            self.remove_bounds_axes()

    def set_view(self, direction):
        """Look toward the structure along lattice axis ``a``, ``b``, or ``c``.

        Args:
            direction: Row index of the requested lattice vector.

        Return:
            None. Camera distance is retained where possible.
        """
        if len(self.lattices) == 0:
            return
        vector = np.asarray(self.lattices[self.current_index][direction], dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return
        vector /= norm
        bounds = self._scene_bounds(
            np.asarray(self.coordinates[self.current_index], dtype=float),
            np.asarray(self.lattices[self.current_index], dtype=float),
        )
        center = np.array(
            [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
        )
        distance = float(np.linalg.norm(np.asarray(self.camera.position) - np.asarray(self.camera.focal_point)))
        if distance == 0:
            distance = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 2.5
        up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(vector, up))) > 0.95:
            up = np.array([0.0, 1.0, 0.0])
        self.camera.focal_point = center
        self.camera.position = center + vector * distance
        self.camera.up = up
        self._capture_zoom_reference()
        self.zoom_percent = 100
        self.reset_camera_clipping_range()
        self.render()

    def reset_view(self):
        """Restore an isometric fitted view of the current scene.

        Args:
            None.

        Return:
            None. The structure data and selection are unchanged.
        """
        if len(self.coordinates) == 0:
            return
        self.camera_position = "iso"
        self.reset_camera()
        self._capture_zoom_reference()
        self.zoom_percent = 100
        self.render()

    def _capture_zoom_reference(self):
        """Capture the current camera distance as the 100 percent zoom reference."""
        position = np.asarray(self.camera.position, dtype=float)
        focal_point = np.asarray(self.camera.focal_point, dtype=float)
        self._zoom_base_distance = float(np.linalg.norm(position - focal_point))

    def set_zoom_percent(self, percent):
        """Set camera zoom relative to the current 100 percent reference.

        Args:
            percent: Zoom percentage from 10 to 1000.

        Return:
            None. The camera is moved without rebuilding scene geometry.
        """
        if self._zoom_base_distance is None:
            self._capture_zoom_reference()
        self.zoom_percent = int(np.clip(float(percent), 10.0, 1000.0))
        focal_point = np.asarray(self.camera.focal_point, dtype=float)
        position = np.asarray(self.camera.position, dtype=float)
        direction = position - focal_point
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm == 0.0 or self._zoom_base_distance is None:
            return
        direction /= direction_norm
        self.camera.position = focal_point + direction * self._zoom_base_distance * 100.0 / self.zoom_percent
        self.reset_camera_clipping_range()
        self.render()

    def set_axes_visible(self, visible):
        """Set orientation-axes visibility.

        Args:
            visible: Requested visibility state.

        Return:
            None. The current scene is redrawn.
        """
        self.axes_visible = bool(visible)
        self.draw_frame()

    def set_grid_visible(self, visible):
        """Set world-coordinate grid visibility.

        Args:
            visible: Requested visibility state.

        Return:
            None. The current scene is redrawn.
        """
        self.grid_visible = bool(visible)
        self.draw_frame()

    def set_bonds(self, enabled):
        """Enable or disable Euclidean covalent-radius bonds.

        Args:
            enabled: Requested bond visibility.

        Return:
            None. The current scene is redrawn.
        """
        self.bonds_enabled = bool(enabled)
        self.draw_frame()

    def set_bond_tolerance(self, tolerance):
        """Set the multiplicative covalent-radius bond cutoff.

        Args:
            tolerance: Nonnegative cutoff multiplier.

        Return:
            None. Visible bonds are redrawn immediately.
        """
        self.bond_tolerance = max(0.0, float(tolerance))
        if self.bonds_enabled:
            self.draw_frame()

    def set_bond_width(self, width):
        """Set the physical bond-cylinder radius in angstroms.

        Args:
            width: Positive cylinder radius.

        Return:
            None. Visible bonds are redrawn immediately.
        """
        self.bond_width = max(0.001, float(width))
        if self.bonds_enabled:
            self.draw_frame()

    def set_bond_alpha(self, alpha):
        """Set bond-cylinder opacity.

        Args:
            alpha: Opacity in the closed interval from zero to one.

        Return:
            None. Visible bonds are redrawn immediately.
        """
        self.bond_alpha = float(np.clip(alpha, 0.0, 1.0))
        if self.bonds_enabled:
            self.draw_frame()

    def set_atom_styles(self, styles):
        """Replace per-element display styles.

        Args:
            styles: Mapping from element symbol to style settings.

        Return:
            None. The current frame is redrawn.
        """
        self.atom_styles = dict(styles)
        self.draw_frame()

    def set_appearance(
        self,
        styles,
        material,
        lighting,
        edges,
        edge_color,
        edge_width,
        atom_style=None,
        custom_radius=None,
        background=None,
        ambient=None,
        diffuse=None,
        specular=None,
        roughness=None,
        metallic=None,
        axes=None,
        grid=None,
    ):
        """Apply element and global appearance settings in one redraw.

        Args:
            styles: Mapping from element symbols to per-element settings.
            material: Global VTK sphere material preset.
            lighting: Whether atom meshes use scene lighting.
            edges: Whether atom mesh boundaries are visible.
            edge_color: Hexadecimal atom-boundary color.
            edge_width: Atom-boundary line width.
            atom_style: Global atom display mode.
            custom_radius: Global custom sphere radius.
            background: Hexadecimal renderer background color.
            ambient: Ambient material coefficient.
            diffuse: Diffuse material coefficient.
            specular: Specular material coefficient.
            roughness: PBR roughness coefficient.
            metallic: PBR metallic coefficient.
            axes: Whether orientation axes are visible.
            grid: Whether the coordinate grid is visible.

        Return:
            None. All appearance settings are validated and rendered together.
        """
        valid_materials = {"default", "matte", "glossy", "metallic"}
        if material not in valid_materials:
            raise ValueError(f"unsupported atom material: {material}")
        self.atom_styles = dict(styles)
        material_changed = material != self.atom_material
        self.atom_material = material
        if material_changed:
            self._set_material_preset(material)
        self.atom_lighting = bool(lighting)
        self.atom_edges = bool(edges)
        self.atom_edge_color = str(edge_color)
        self.atom_edge_width = max(0.1, float(edge_width))
        if atom_style is not None:
            if atom_style not in {"point", "covalent", "vdw", "custom"}:
                raise ValueError(f"unsupported atom display style: {atom_style}")
            self.default_atom_style = atom_style
        if custom_radius is not None:
            self.default_custom_radius = max(0.01, float(custom_radius))
        if background is not None:
            self._background_color_value = self._normalize_color(background)
            self.set_background(self._background_color_value)
        if ambient is not None:
            self.material_ambient = float(np.clip(ambient, 0.0, 1.0))
        if diffuse is not None:
            self.material_diffuse = float(np.clip(diffuse, 0.0, 1.0))
        if specular is not None:
            self.material_specular = float(np.clip(specular, 0.0, 1.0))
        if roughness is not None:
            self.material_roughness = float(np.clip(roughness, 0.0, 1.0))
        if metallic is not None:
            self.material_metallic = float(np.clip(metallic, 0.0, 1.0))
        if axes is not None:
            self.axes_visible = bool(axes)
        if grid is not None:
            self.grid_visible = bool(grid)
        self.draw_frame()

    def set_measurement_mode(self, mode):
        """Enter or leave the two-atom or three-atom measurement selector.

        Args:
            mode: ``"length"``, ``"angle"``, or ``None``.

        Return:
            None. A new mode clears any incomplete temporary selection.
        """
        if mode not in {None, "length", "angle"}:
            raise ValueError(f"unsupported measurement mode: {mode}")
        self.measurement_mode = mode
        self.measurement_selection = []
        self.measurementModeChanged.emit(mode)

    def set_background_color(self, color):
        """Set the renderer background color and redraw the current frame.

        Args:
            color: A Qt/VTK-compatible color string.

        Return:
            None. The current scene is redrawn.
        """
        self._background_color_value = self._normalize_color(color)
        self.set_background(self._background_color_value)
        self.draw_frame()

    @staticmethod
    def _normalize_color(color):
        """Normalize Qt, VTK, and PyVista colors to a hexadecimal string.

        Args:
            color: Color value accepted by the renderer.

        Return:
            A ``#rrggbb`` string, or a string representation as a fallback.
        """
        if isinstance(color, str):
            return color
        try:
            normalized = pv.Color(color).hex_rgb
            if isinstance(normalized, str):
                return normalized
        except (TypeError, ValueError):
            pass
        name = getattr(color, "name", None)
        if callable(name):
            value = name()
            if isinstance(value, str):
                return value
        return str(color)

    @staticmethod
    def default_metric_style():
        """Return one detached default measurement style dictionary.

        Args:
            None.

        Return:
            Default yellow dashed line and label settings.
        """
        return {
            "line_color": "#FFFF00",
            "text_color": "#FFFF00",
            "line_style": "dashed",
            "font_size": 14,
        }

    def _draw_measurements(self, positions):
        """Draw persistent dashed distance and angle annotations for this frame.

        Args:
            positions: Current frame Cartesian coordinates.

        Return:
            None. Measurement values in both dictionaries are refreshed in place.
        """
        for pair in self.length_measurements:
            first, second = sorted(pair)
            if max(first, second) >= len(positions):
                continue
            first_point = np.asarray(positions[first], dtype=float)
            second_point = np.asarray(positions[second], dtype=float)
            metric = self.length_measurements[pair]
            style = {**self.default_metric_style(), **metric.get("style", {})}
            line = pv.Line(first_point, second_point)
            actor = self.add_mesh(line, color=style["line_color"], line_width=2.0, lighting=False, render=False)
            self.measurement_actors.append(actor)
            if style["line_style"] == "dashed":
                actor.GetProperty().SetLineStipplePattern(0xF0F0)
                actor.GetProperty().SetLineStippleRepeatFactor(1)
            midpoint = (first_point + second_point) / 2.0
            value = distance_between(positions, first, second)
            metric["value"] = value
            label_actor = self.add_point_labels(
                pv.PolyData(np.asarray([midpoint])),
                [f"{value:.2f}"],
                font_size=int(style["font_size"]),
                text_color=style["text_color"],
                shape=None,
                show_points=False,
                always_visible=True,
                render=False,
            )
            self.measurement_actors.append(label_actor)
        for vertex, endpoints in self.angle_measurements:
            first, second = sorted(endpoints)
            if max(vertex, first, second) >= len(positions):
                continue
            metric = self.angle_measurements[(vertex, endpoints)]
            style = {**self.default_metric_style(), **metric.get("style", {})}
            vertex_point = np.asarray(positions[vertex], dtype=float)
            first_point = np.asarray(positions[first], dtype=float)
            second_point = np.asarray(positions[second], dtype=float)
            for endpoint in (first_point, second_point):
                actor = self.add_mesh(
                    pv.Line(vertex_point, endpoint),
                    color=style["line_color"],
                    line_width=2.0,
                    lighting=False,
                    render=False,
                )
                self.measurement_actors.append(actor)
                if style["line_style"] == "dashed":
                    actor.GetProperty().SetLineStipplePattern(0xF0F0)
                    actor.GetProperty().SetLineStippleRepeatFactor(1)
            direction = (first_point - vertex_point) + (second_point - vertex_point)
            direction_norm = float(np.linalg.norm(direction))
            label_position = vertex_point + direction / direction_norm * min(
                np.linalg.norm(first_point - vertex_point),
                np.linalg.norm(second_point - vertex_point),
            ) * 0.45 if direction_norm > 0 else vertex_point
            try:
                value = angle_between(positions, vertex, first, second)
            except ValueError:
                continue
            metric["value"] = value
            label_actor = self.add_point_labels(
                pv.PolyData(np.asarray([label_position])),
                [f"{value:.2f} deg"],
                font_size=int(style["font_size"]),
                text_color=style["text_color"],
                shape=None,
                show_points=False,
                always_visible=True,
                render=False,
            )
            self.measurement_actors.append(label_actor)

    def begin_playback(self):
        """Enter lightweight animation mode without rebuilding the scene.

        Args:
            None.

        Return:
            None. Dynamic bond, lattice, and measurement actors are hidden until playback ends.
        """
        if self.playback_active:
            return
        if not (0 <= self.current_index < len(self.coordinates)):
            return
        self.playback_active = True
        self._playback_base_positions = np.asarray(
            self.coordinates[self.current_index], dtype=float
        ).copy()
        self._playback_base_elements = np.asarray(
            self.elements[self.current_index], dtype=str
        ).copy()
        for actor in (
            self.bond_actors
            + self.lattice_actors
            + self.measurement_actors
            + self.selection_shell_actors
        ):
            actor.SetVisibility(False)

    def update_frame_preview(self, index):
        """Move existing atom glyph meshes to one compatible frame.

        Args:
            index: Frame index to preview.

        Return:
            ``True`` when the existing scene was updated, otherwise ``False``.
        """
        if not self.playback_active or not 0 <= index < len(self.coordinates):
            return False
        positions = np.asarray(self.coordinates[index], dtype=float)
        elements = np.asarray(self.elements[index], dtype=str)
        base_positions = self._playback_base_positions
        base_elements = self._playback_base_elements
        if base_positions is None or base_elements is None:
            return False
        if len(positions) != len(base_positions) or not np.array_equal(
            elements, base_elements
        ):
            return False
        for mesh, indices, base_points in self.atom_meshes:
            if len(indices) == 0 or base_points.shape[0] % len(indices) != 0:
                return False
            displacement = positions[indices] - base_positions[indices]
            repeats = base_points.shape[0] // len(indices)
            mesh.points = base_points + np.repeat(displacement, repeats, axis=0)
            mesh.GetPoints().Modified()
            mesh.Modified()
        self.current_index = index
        if 0 <= index < len(self.frame_metrics):
            self.length_measurements = self.frame_metrics[index]["length"]
            self.angle_measurements = self.frame_metrics[index]["angle"]
        self.render()
        return True

    def finish_playback(self):
        """Leave lightweight animation mode and rebuild dynamic geometry once."""
        if not self.playback_active:
            return
        self.playback_active = False
        self._playback_base_positions = None
        self._playback_base_elements = None
        self.draw_frame()

    def set_atom_material(self, material):
        """Set the global VTK sphere material preset.

        Args:
            material: One of ``default``, ``matte``, ``glossy``, or ``metallic``.

        Return:
            None. The current frame is redrawn.
        """
        valid_materials = {"default", "matte", "glossy", "metallic"}
        if material not in valid_materials:
            raise ValueError(f"unsupported atom material: {material}")
        self.atom_material = material
        self._set_material_preset(material)
        self.draw_frame()

    def set_atom_lighting(self, enabled):
        """Enable or disable lighting on atom meshes.

        Args:
            enabled: Requested lighting state.

        Return:
            None. The current frame is redrawn.
        """
        self.atom_lighting = bool(enabled)
        self.draw_frame()

    def set_atom_edges(self, enabled):
        """Enable or disable atom mesh boundaries.

        Args:
            enabled: Requested boundary visibility.

        Return:
            None. The current frame is redrawn.
        """
        self.atom_edges = bool(enabled)
        self.draw_frame()

    def set_atom_edge_style(self, color, width):
        """Set the global atom boundary color and line width.

        Args:
            color: Hexadecimal boundary color.
            width: Positive VTK line width.

        Return:
            None. The current frame is redrawn.
        """
        self.atom_edge_color = str(color)
        self.atom_edge_width = max(0.1, float(width))
        self.draw_frame()

    def set_error_handler(self, handler):
        """Register a concise error sink for recoverable mouse errors.

        Args:
            handler: Callable receiving one human-readable error message.

        Return:
            None. The handler is retained by the viewer.
        """
        self.error_handler = handler

    def _report_interaction_error(self, error):
        """Report a recoverable interaction failure without raising into Qt.

        Args:
            error: Exception raised while processing a mouse gesture.

        Return:
            None. The configured error sink receives a concise message.
        """
        message = f"Interaction error: {error}"
        if self.error_handler is not None:
            self.error_handler(message)

    def _display_position(self, event):
        """Convert a Qt mouse position to VTK display coordinates.

        Args:
            event: Qt mouse event using a top-left coordinate origin.

        Return:
            Two display coordinates using VTK's bottom-left origin.
        """
        position = event.position()
        render_width, render_height = self.ren_win.GetSize()
        scale_x = render_width / max(self.width(), 1)
        scale_y = render_height / max(self.height(), 1)
        return np.array([position.x() * scale_x, render_height - position.y() * scale_y], dtype=float)

    def _project_atoms(self):
        """Project current atom coordinates into VTK display coordinates.

        Args:
            None.

        Return:
            An ``(N, 3)`` array containing display x, y, and depth.
        """
        if not (0 <= self.current_index < len(self.coordinates)):
            return np.empty((0, 3), dtype=float)
        projected = []
        for point in self.coordinates[self.current_index]:
            self.renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
            self.renderer.WorldToDisplay()
            projected.append(self.renderer.GetDisplayPoint())
        return np.asarray(projected, dtype=float)

    def _world_at_display_depth(self, point, depth):
        """Unproject a display position at one retained display depth.

        Args:
            point: Two-dimensional VTK display coordinates.
            depth: VTK display depth captured at drag start.

        Return:
            Cartesian world coordinates corresponding to the display point.
        """
        self.renderer.SetDisplayPoint(float(point[0]), float(point[1]), float(depth))
        self.renderer.DisplayToWorld()
        world = np.asarray(self.renderer.GetWorldPoint(), dtype=float)
        return world[:3] / world[3] if world[3] != 0 else world[:3]

    def _project_point(self, point):
        """Project one Cartesian point into VTK display coordinates.

        Args:
            point: Cartesian position with three components.

        Return:
            Display x, y, and depth coordinates.
        """
        self.renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
        self.renderer.WorldToDisplay()
        return np.asarray(self.renderer.GetDisplayPoint(), dtype=float)

    def _axis_angle_matrix(self, axis, angle):
        """Build a homogeneous rotation matrix around the drag center.

        Args:
            axis: Cartesian rotation axis.
            angle: Signed angle in radians.

        Return:
            A homogeneous matrix suitable for VTK actor preview transforms.
        """
        axis = np.asarray(axis, dtype=float)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm == 0 or self.drag_center is None:
            return np.eye(4)
        axis /= axis_norm
        cosine = np.cos(angle)
        sine = np.sin(angle)
        cross = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        rotation = cosine * np.eye(3) + (1.0 - cosine) * np.outer(axis, axis) + sine * cross
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = self.drag_center - rotation @ self.drag_center
        return matrix

    def _preview_selected(self, matrix):
        """Apply one matrix to selected actors without changing coordinates.

        Args:
            matrix: Homogeneous world-coordinate preview transform.

        Return:
            None. Only lightweight VTK actor state and the canvas are updated.
        """
        transform = vtkTransform()
        transform.SetMatrix(tuple(np.asarray(matrix, dtype=float).ravel()))
        for actor in self.selected_actors:
            actor.SetUserTransform(transform)
        self.drag_preview_matrix = np.asarray(matrix, dtype=float)
        self.render()

    def _camera_mouse_event(self, event, button, buttons):
        """Remap a right-button Qt event to VTK's native left-button orbit.

        Args:
            event: Original Qt mouse event.
            button: Button associated with this event transition.
            buttons: Buttons considered held after the transition.

        Return:
            A Qt mouse event suitable for the native VTK trackball handler.
        """
        return QMouseEvent(
            event.type(),
            event.position(),
            event.scenePosition(),
            event.globalPosition(),
            button,
            buttons,
            event.modifiers(),
            event.pointingDevice(),
        )

    def begin_atom_addition(self, symbol):
        """Enter temporary reference-based atom addition mode.

        Args:
            symbol: Validated chemical element symbol for new atoms.

        Returns:
            None. No retained structure arrays are changed.
        """
        self.cancel_atom_addition()
        self.add_mode = True
        self.add_symbol = str(symbol)
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        self._log_add_status("Add mode: click a reference atom, then place atoms; Enter commits, Escape cancels.")

    def _log_add_status(self, message):
        """Send a temporary atom-addition message to the configured error sink.

        Args:
            message: Human-readable interaction status.

        Returns:
            None. Status is omitted when no GUI log sink is configured.
        """
        if self.error_handler is not None:
            self.error_handler(message)

    def _remove_add_actors(self):
        """Remove all temporary actors used by reference-based addition.

        Args:
            None.

        Returns:
            None. The retained raw lists are untouched.
        """
        for actor in self.add_fixed_actors:
            self.remove_actor(actor, render=False)
        if self.add_active_actor is not None:
            self.remove_actor(self.add_active_actor, render=False)
        self.add_fixed_actors = []
        self.add_active_actor = None

    def cancel_atom_addition(self):
        """Cancel temporary atom addition and restore normal selection mode.

        Args:
            None.

        Returns:
            None. Any temporary actor and pending atom are discarded.
        """
        self._remove_add_actors()
        self.add_mode = False
        self.add_symbol = None
        self.add_reference_position = None
        self.add_reference_symbol = None
        self.add_pending_elements = []
        self.add_pending_positions = []
        self._add_preview_position = None
        self.render()

    def _finish_atom_addition(self):
        """Commit all fixed temporary atoms through one viewer signal.

        Args:
            None.

        Returns:
            None. MainWindow receives the complete pending chain atomically.
        """
        if len(self.add_pending_positions) == 0:
            self.cancel_atom_addition()
            return
        frame_index = int(self.current_index)
        symbols = np.asarray(self.add_pending_elements, dtype="<U3")
        positions = np.asarray(self.add_pending_positions, dtype=float)
        self._remove_add_actors()
        self.add_mode = False
        self.add_symbol = None
        self.add_reference_position = None
        self.add_reference_symbol = None
        self.add_pending_elements = []
        self.add_pending_positions = []
        self._add_preview_position = None
        self.atomsAdded.emit(frame_index, symbols, positions)

    def _add_preview_actor(self, position, fixed=False):
        """Create one temporary sphere actor for an atom preview.

        Args:
            position: Cartesian position for the actor.
            fixed: Whether the actor should remain after the next click.

        Returns:
            The created VTK actor.
        """
        symbol = self.add_symbol or "C"
        radius = self._style_radius(symbol)
        source = pv.Sphere(radius=radius, theta_resolution=20, phi_resolution=14)
        actor = self.add_mesh(
            source,
            color=self._style_for(symbol)["color"],
            opacity=0.72 if not fixed else 1.0,
            smooth_shading=True,
            lighting=self.atom_lighting,
            render=False,
        )
        actor.SetPosition(float(position[0]), float(position[1]), float(position[2]))
        if fixed:
            self.add_fixed_actors.append(actor)
        else:
            self.add_active_actor = actor
        return actor

    def _update_add_preview(self, point):
        """Position the temporary atom at a fixed covalent bond length.

        Args:
            point: Two-dimensional VTK display position.

        Returns:
            None. Only temporary actor geometry is changed.
        """
        if not self.add_mode or self.add_reference_position is None:
            return
        reference = np.asarray(self.add_reference_position, dtype=float)
        reference_display = self._project_point(reference)
        plane_world = self._world_at_display_depth(point, reference_display[2]) - reference
        plane_distance = float(np.linalg.norm(plane_world))
        reference_symbol = self.add_reference_symbol or str(self.elements[self.current_index][0])
        bond_length = covalent_radius(reference_symbol) + covalent_radius(self.add_symbol)
        if plane_distance >= bond_length:
            position = reference + plane_world / plane_distance * bond_length
        else:
            _, _, view = self._camera_basis()
            toward_viewer = -view
            position = reference + plane_world
            position += toward_viewer * np.sqrt(max(bond_length * bond_length - plane_distance * plane_distance, 0.0))
        if self.add_active_actor is None:
            self._add_preview_actor(position, fixed=False)
        else:
            self.add_active_actor.SetPosition(float(position[0]), float(position[1]), float(position[2]))
        self._add_preview_position = np.asarray(position, dtype=float)
        self.render()

    def _handle_add_click(self, point):
        """Handle one left click in temporary atom addition mode.

        Args:
            point: Two-dimensional VTK display position.

        Returns:
            None. The first click chooses a reference; later clicks fix previews.
        """
        if self.add_reference_position is None:
            picked = self._pick_atom(point)
            if len(picked) == 0:
                return
            self.add_reference_position = np.array(
                self.coordinates[self.current_index][int(picked[0])], copy=True
            )
            self.add_reference_symbol = str(self.elements[self.current_index][int(picked[0])])
            self._update_add_preview(point)
            return
        if self._add_preview_position is None:
            self._update_add_preview(point)
        if self._add_preview_position is None:
            return
        fixed_position = np.array(self._add_preview_position, copy=True)
        self.add_pending_elements.append(self.add_symbol)
        self.add_pending_positions.append(fixed_position)
        if self.add_active_actor is not None:
            self.add_active_actor.GetProperty().SetOpacity(1.0)
            self.add_fixed_actors.append(self.add_active_actor)
            self.add_active_actor = None
        self.add_reference_position = fixed_position
        self.add_reference_symbol = self.add_symbol
        self._add_preview_position = None
        self._update_add_preview(point)

    def keyPressEvent(self, event):
        """Handle deletion and Enter/Escape controls for atom addition.

        Args:
            event: Qt keyboard event.

        Returns:
            None. The event is consumed for supported structure operations.
        """
        if event.key() == Qt.Key.Key_Escape and self.add_mode:
            self.cancel_atom_addition()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.add_mode:
            self._finish_atom_addition()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Delete and not self.add_mode:
            if len(self.selected_indices) > 0:
                self.deleteRequested.emit(int(self.current_index), self.selected_indices.copy())
            event.accept()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        """Accept local file drops over the OpenGL viewer surface.

        Args:
            event: Qt drag-enter event.

        Returns:
            None. Local URL drops are accepted for MainWindow processing.
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event):
        """Forward local dropped paths to the main window.

        Args:
            event: Qt drop event.

        Returns:
            None. Non-local or empty drops are delegated to Qt.
        """
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def mousePressEvent(self, event):
        """Start camera, selection, translation, or rotation interaction.

        Args:
            event: Qt mouse press event.

        Return:
            None. Gesture state is captured without invoking VTK's default bindings.
        """
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            super().mousePressEvent(event)
            return
        self.pressed_button = event.button()
        self.press_position = self._display_position(event)
        self.press_modifiers = event.modifiers()
        self.dragged = False
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if self.add_mode and event.button() == Qt.MouseButton.LeftButton:
            self.drag_mode = "add_atom"
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            shift = bool(self.press_modifiers & Qt.KeyboardModifier.ShiftModifier)
            alt = bool(self.press_modifiers & Qt.KeyboardModifier.AltModifier)
            self.drag_mode = "rotate_atoms" if shift and alt else "translate_atoms" if shift else "rotate_view"
            has_frame = 0 <= self.current_index < len(self.coordinates)
            has_valid_selection = (
                has_frame
                and len(self.selected_indices) > 0
                and bool(np.all(self.selected_indices >= 0))
                and bool(np.all(self.selected_indices < len(self.coordinates[self.current_index])))
            )
            if self.drag_mode in ("translate_atoms", "rotate_atoms") and has_valid_selection:
                frame_coordinates = self.coordinates[self.current_index]
                self.drag_start_coordinates = frame_coordinates.copy()
                self.drag_start_positions = frame_coordinates[self.selected_indices].copy()
                self.drag_start_mouse = self.press_position.copy()
                self.drag_start_basis = self._camera_basis()
                self.drag_center = self.drag_start_positions.mean(axis=0)
                center_display = self._project_point(self.drag_center)
                self.drag_center_screen = center_display[:2]
                self.drag_start_depth = center_display[2]
                self.drag_start_world = self._world_at_display_depth(self.drag_start_mouse, self.drag_start_depth)
                self.drag_last_mouse = self.drag_start_mouse.copy()
                self.drag_outer_angle = 0.0
                self.drag_preview_matrix = np.eye(4)
                render_width, render_height = self.ren_win.GetSize()
                inner_half_width = min(render_width, render_height) * 0.25
                offset = np.abs(self.drag_start_mouse - self.drag_center_screen)
                self.drag_rotation_zone = "inner" if np.all(offset <= inner_half_width) else "outer"
                for actor in self.bond_actors:
                    actor.SetVisibility(False)
                self.render()
            elif self.drag_mode in ("translate_atoms", "rotate_atoms") and len(self.selected_indices) > 0:
                self.selected_indices = np.empty(0, dtype=int)
                self._report_interaction_error(
                    ValueError("the atom selection no longer matches the current frame")
                )
            elif self.drag_mode == "rotate_view":
                camera_event = self._camera_mouse_event(
                    event,
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.LeftButton,
                )
                super().mousePressEvent(camera_event)
        else:
            self.drag_mode = "select"
            self.selection_start = event.position().toPoint()
            self.selection_band.setGeometry(QRect(self.selection_start, self.selection_start))
            self.selection_band.show()
        event.accept()

    def mouseMoveEvent(self, event):
        """Preview the active mouse gesture using camera or world coordinates.

        Args:
            event: Qt mouse move event.

        Return:
            None. Selected actors move during preview; coordinates remain unchanged.
        """
        current = self._display_position(event)
        if self.add_mode and self.pressed_button == Qt.MouseButton.NoButton:
            self._update_add_preview(current)
            event.accept()
            return
        if self.pressed_button == Qt.MouseButton.NoButton:
            super().mouseMoveEvent(event)
            return
        if np.linalg.norm(current - self.press_position) > 4:
            self.dragged = True
        has_atom_drag_state = (
            self.drag_start_mouse is not None
            and self.drag_start_basis is not None
            and self.drag_start_depth is not None
            and self.drag_start_world is not None
        )
        if self.drag_mode == "rotate_view" and self.dragged:
            camera_event = self._camera_mouse_event(
                event,
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
            )
            super().mouseMoveEvent(camera_event)
            if self.add_mode:
                self._update_add_preview(current)
        elif self.add_mode and self.drag_mode == "add_atom":
            self._update_add_preview(current)
        elif self.drag_mode == "select" and self.selection_start is not None:
            self.selection_band.setGeometry(
                QRect(self.selection_start, event.position().toPoint()).normalized()
            )
        elif (
            self.drag_mode in ("translate_atoms", "rotate_atoms")
            and self.dragged
            and len(self.selected_indices) > 0
            and has_atom_drag_state
        ):
            total_delta = current - self.drag_start_mouse
            if self.drag_mode == "translate_atoms":
                current_world = self._world_at_display_depth(current, self.drag_start_depth)
                displacement = current_world - self.drag_start_world
                matrix = np.eye(4)
                matrix[:3, 3] = displacement
            else:
                right, up, view = self.drag_start_basis
                if self.drag_rotation_zone == "inner":
                    angle_y = -total_delta[1] * 0.012
                    angle_z = total_delta[0] * 0.012
                    rotation_y = self._axis_angle_matrix(right, angle_y)
                    rotation_z = self._axis_angle_matrix(up, angle_z)
                    matrix = rotation_z @ rotation_y
                else:
                    previous = self.drag_last_mouse - self.drag_center_screen
                    current_offset = current - self.drag_center_screen
                    cross = previous[0] * current_offset[1] - previous[1] * current_offset[0]
                    dot = float(np.dot(previous, current_offset))
                    self.drag_outer_angle += np.arctan2(cross, dot)
                    matrix = self._axis_angle_matrix(-view, self.drag_outer_angle)
                    self.drag_last_mouse = current.copy()
            try:
                self._preview_selected(matrix)
            except (IndexError, RuntimeError, TypeError, ValueError) as error:
                self._report_interaction_error(error)
        event.accept()

    def mouseReleaseEvent(self, event):
        """Finish the active gesture and commit one coordinate edit if needed.

        Args:
            event: Qt mouse release event.

        Return:
            None. Coordinates and bonds are updated once before drag state is cleared.
        """
        finished_drag_mode = self.drag_mode
        try:
            if event.button() != self.pressed_button:
                super().mouseReleaseEvent(event)
                return
            if self.add_mode and self.pressed_button == Qt.MouseButton.LeftButton:
                self._handle_add_click(self._display_position(event))
                self.pressed_button = Qt.MouseButton.NoButton
                self.drag_mode = None
                self.dragged = False
                event.accept()
                return
            if self.pressed_button == Qt.MouseButton.LeftButton:
                self.selection_band.hide()
                self.selection_start = None
                additive = bool(self.press_modifiers & Qt.KeyboardModifier.ControlModifier)
                finish = self._display_position(event)
                if len(self.coordinates) == 0:
                    self._update_selection(np.empty(0, dtype=int), additive=False)
                elif self.measurement_mode is not None:
                    if not self.dragged:
                        picked = self._pick_atom(finish)
                        if len(picked) == 0:
                            self.set_measurement_mode(None)
                        else:
                            atom_index = int(picked[0])
                            if atom_index not in self.measurement_selection:
                                self.measurement_selection.append(atom_index)
                            expected_count = 2 if self.measurement_mode == "length" else 3
                            if len(self.measurement_selection) == expected_count:
                                if self.measurement_mode == "length":
                                    key = frozenset(self.measurement_selection)
                                    if len(key) == 2:
                                        if key in self.length_measurements:
                                            del self.length_measurements[key]
                                        else:
                                            self.length_measurements[key] = {
                                                "value": distance_between(
                                                    self.coordinates[self.current_index],
                                                    self.measurement_selection[0],
                                                    self.measurement_selection[1],
                                                ),
                                                "style": self.default_metric_style(),
                                            }
                                else:
                                    vertex = self.measurement_selection[1]
                                    endpoints = frozenset(
                                        (self.measurement_selection[0], self.measurement_selection[2])
                                    )
                                    if len(endpoints) == 2:
                                        key = (vertex, endpoints)
                                        if key in self.angle_measurements:
                                            del self.angle_measurements[key]
                                        else:
                                            self.angle_measurements[key] = {
                                                "value": angle_between(
                                                    self.coordinates[self.current_index],
                                                    vertex,
                                                    self.measurement_selection[0],
                                                    self.measurement_selection[2],
                                                ),
                                                "style": self.default_metric_style(),
                                            }
                                self.set_measurement_mode(None)
                elif self.dragged:
                    screen = self._project_atoms()
                    inside = (
                        (screen[:, 0] >= min(self.press_position[0], finish[0]))
                        & (screen[:, 0] <= max(self.press_position[0], finish[0]))
                        & (screen[:, 1] >= min(self.press_position[1], finish[1]))
                        & (screen[:, 1] <= max(self.press_position[1], finish[1]))
                    )
                    self._update_selection(np.flatnonzero(inside), additive)
                else:
                    self._update_selection(self._pick_atom(finish), additive)
            elif self.drag_mode == "rotate_view":
                camera_event = self._camera_mouse_event(
                    event,
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.NoButton,
                )
                super().mouseReleaseEvent(camera_event)
            should_commit = (
                finished_drag_mode in ("translate_atoms", "rotate_atoms")
                and self.dragged
                and self.drag_start_coordinates is not None
                and self.drag_start_positions is not None
                and len(self.selected_indices) > 0
            )
            if should_commit:
                frame_coordinates = self.coordinates[self.current_index]
                if frame_coordinates.shape != self.drag_start_coordinates.shape:
                    raise ValueError("current frame changed shape during atom manipulation")
                rotation = self.drag_preview_matrix[:3, :3]
                translation = self.drag_preview_matrix[:3, 3]
                transformed = self.drag_start_positions @ rotation.T + translation
                frame_coordinates[self.selected_indices] = transformed
                after_coordinates = frame_coordinates.copy()
                if not np.array_equal(self.drag_start_coordinates, after_coordinates):
                    self.coordinatesEdited.emit(
                        self.current_index,
                        self.drag_start_coordinates,
                        after_coordinates,
                    )
        except (IndexError, RuntimeError, TypeError, ValueError) as error:
            self._report_interaction_error(error)

        # Section: release cleanup. Rebuild once so committed atoms and bonds agree.
        self.pressed_button = Qt.MouseButton.NoButton
        self.drag_mode = None
        self.dragged = False
        self.drag_start_coordinates = None
        self.drag_start_positions = None
        self.drag_start_mouse = None
        self.drag_start_basis = None
        self.drag_start_depth = None
        self.drag_start_world = None
        self.drag_center = None
        self.drag_center_screen = None
        self.drag_rotation_zone = None
        self.drag_last_mouse = None
        self.drag_outer_angle = 0.0
        self.drag_preview_matrix = np.eye(4)
        if finished_drag_mode not in ("rotate_view", "select"):
            try:
                self.draw_frame()
            except (IndexError, RuntimeError, TypeError, ValueError) as error:
                self._report_interaction_error(error)
        event.accept()

    def _pick_atom(self, point):
        """Find the closest projected atom within the click tolerance.

        Args:
            point: Two-dimensional VTK display coordinates.

        Return:
            One atom index or an empty integer array for blank space.
        """
        screen = self._project_atoms()
        if not len(screen):
            return np.empty(0, dtype=int)
        distances = np.linalg.norm(screen[:, :2] - point, axis=1)
        right, _, _ = self._camera_basis()
        projected_radii = []
        for index, (position, symbol) in enumerate(
            zip(self.coordinates[self.current_index], self.elements[self.current_index])
        ):
            edge = np.asarray(position, dtype=float) + right * self._style_radius(
                str(symbol), index
            )
            self.renderer.SetWorldPoint(float(edge[0]), float(edge[1]), float(edge[2]), 1.0)
            self.renderer.WorldToDisplay()
            edge_screen = np.asarray(self.renderer.GetDisplayPoint())[:2]
            projected_radii.append(np.linalg.norm(edge_screen - screen[index, :2]))
        hit = distances <= np.maximum(8.0, np.asarray(projected_radii))
        candidates = np.flatnonzero(hit)
        if not len(candidates):
            return np.empty(0, dtype=int)
        nearest = candidates[np.argmin(screen[candidates, 2])]
        return np.array([nearest], dtype=int)

    def _update_selection(self, picked, additive):
        """Replace or toggle selected atom indices.

        Args:
            picked: Atom indices found by click or box selection.
            additive: Whether Ctrl requests toggle selection.

        Return:
            None. Selection state is changed without wrapping structure data.
        """
        if additive:
            selected = set(self.selected_indices.tolist())
            for index in picked.tolist():
                if index in selected:
                    selected.remove(index)
                else:
                    selected.add(index)
            self.selected_indices = np.asarray(sorted(selected), dtype=int)
        else:
            self.selected_indices = picked.astype(int, copy=True)
        self.refresh_selection_overlay()
        self.selectionChanged.emit(self.selected_indices.copy())

    def refresh_selection_overlay(self):
        """Refresh selected atom actors while preserving the camera state.

        Args:
            None.

        Return:
            None. Camera and zoom-reference state are untouched.
        """
        self.draw_frame(reset_camera=False)

    def _camera_basis(self):
        """Return camera-right, camera-up, and viewing unit vectors.

        Args:
            None.

        Return:
            Three Cartesian unit vectors based on the active VTK camera.
        """
        view = np.asarray(self.camera.GetDirectionOfProjection(), dtype=float)
        view /= np.linalg.norm(view)
        up = np.asarray(self.camera.GetViewUp(), dtype=float)
        up /= np.linalg.norm(up)
        right = np.cross(view, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, view)
        up /= np.linalg.norm(up)
        return right, up, view
