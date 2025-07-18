import open3d as o3d
import numpy as np
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

class BoundingBoxApp:
    def __init__(self, pcd_path):
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "Bounding Box Cropper", 1024, 768)

        # 기본 스타일 값들
        self.point_color = [0.0, 1.0, 0.0]    # 흰색
        self.box_color   = [1.0, 0.0, 0.0]    # 빨간색
        self.point_size  = int(2 * self.window.scaling)

        # SceneWidget
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget)

        # 버튼들
        self.show_btn = gui.Button("Show Cropped Cloud")
        self.show_btn.enabled = False
        self.show_btn.set_on_clicked(self.show_cropped)
        self.window.add_child(self.show_btn)

        self.export_btn = gui.Button("Export Cropped Cloud")
        self.export_btn.enabled = False
        self.export_btn.set_on_clicked(self.export_cropped)
        self.window.add_child(self.export_btn)

        # 새로 추가된 UI 컨트롤들
        self.size_slider = gui.Slider(gui.Slider.INT)
        self.size_slider.set_limits(1, 20)
        self.size_slider.int_value = self.point_size
        self.size_slider.set_on_value_changed(self.on_size_changed)
        self.window.add_child(self.size_slider)

        self.point_color_edit = gui.ColorEdit()
        self.point_color_edit.color_value = gui.Color(
            self.point_color[0], self.point_color[1], self.point_color[2], 1.0
        )  # :contentReference[oaicite:0]{index=0}
        self.point_color_edit.set_on_value_changed(self.on_point_color_changed)
        self.window.add_child(self.point_color_edit)

        self.box_color_edit = gui.ColorEdit()
        self.box_color_edit.color_value = gui.Color(
            self.box_color[0], self.box_color[1], self.box_color[2], 1.0
        )  # :contentReference[oaicite:1]{index=1}
        self.box_color_edit.set_on_value_changed(self.on_box_color_changed)
        self.window.add_child(self.box_color_edit)

        # 레이아웃 콜백
        self.window.set_on_layout(self._on_layout)
        self._on_layout(None)  # 초기 한번 강제 호출

        # PointCloud 로드
        self.pcd = o3d.io.read_point_cloud(pcd_path)
        print(f"[DEBUG] Loaded point cloud with {len(self.pcd.points)} points")
        self._add_pcd_geometry()

        # 카메라 셋업
        bounds = self.pcd.get_axis_aligned_bounding_box()
        self.widget.setup_camera(60, bounds, bounds.get_center())

        # 마우스 클릭
        self.ctrl_clicks = []
        self.widget.set_on_mouse(self.on_mouse)

    def _on_layout(self, _):
        r = self.window.content_rect
        margin = 10
        btn_w, btn_h = 160, 30

        # SceneWidget 전체
        self.widget.frame = r

        # 첫 번째 줄: Show/Export 버튼
        self.show_btn.frame   = gui.Rect(r.x + margin, r.y + margin, btn_w, btn_h)
        self.export_btn.frame = gui.Rect(r.x + margin + btn_w + margin,
                                         r.y + margin, btn_w, btn_h)

        # 두 번째 줄: Point Size Slider
        y2 = r.y + margin + btn_h + margin
        self.size_slider.frame = gui.Rect(r.x + margin, y2, btn_w*2 + margin, btn_h)

        # 세 번째/네 번째 줄: 색상 편집기
        y3 = y2 + btn_h + margin
        self.point_color_edit.frame = gui.Rect(r.x + margin, y3, btn_w*2 + margin, btn_h)
        y4 = y3 + btn_h + margin
        self.box_color_edit.frame   = gui.Rect(r.x + margin, y4, btn_w*2 + margin, btn_h)

    def _add_pcd_geometry(self):
        """현재 point_color, point_size 를 반영하여 pointcloud 그리기"""
        if self.widget.scene.has_geometry("pcd"):
            self.widget.scene.remove_geometry("pcd")
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = (*self.point_color, 1.0)
        mat.point_size = self.point_size
        self.widget.scene.add_geometry("pcd", self.pcd, mat)

    def _update_cropped_geometry(self):
        """이미 show_cropped() 를 눌러서 cropped 가 씬에 있으면 갱신"""
        if self.widget.scene.has_geometry("cropped"):
            self.widget.scene.remove_geometry("cropped")
            cropped = self.pcd.crop(self.aabb)
            mat = rendering.MaterialRecord()
            mat.shader = "defaultUnlit"
            mat.base_color = (*self.point_color, 1.0)
            mat.point_size = self.point_size
            self.widget.scene.add_geometry("cropped", cropped, mat)

    def _add_box_geometry(self):
        """현재 box_color 를 반영하여 AABB 테두리 그리기"""
        if self.widget.scene.has_geometry("bbox"):
            self.widget.scene.remove_geometry("bbox")
        box_ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(self.aabb)
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.base_color = (*self.box_color, 1.0)
        mat.line_width = max(1.0, 2.0 * self.window.scaling)
        self.widget.scene.add_geometry("bbox", box_ls, mat)

    def on_mouse(self, event):
        if (event.type == gui.MouseEvent.Type.BUTTON_DOWN and
            event.is_button_down(gui.MouseButton.LEFT) and
            event.is_modifier_down(gui.KeyModifier.CTRL)):

            def depth_cb(depth_image):
                x = int(event.x - self.widget.frame.x)
                y = int(event.y - self.widget.frame.y)
                depth = np.asarray(depth_image)[y, x]
                if depth >= 1.0: return
                world = self.widget.scene.camera.unproject(
                    x, y, float(depth),
                    self.widget.frame.width, self.widget.frame.height)
                self.ctrl_clicks.append(np.array(world))
                if len(self.ctrl_clicks) == 2:
                    self.aabb = o3d.geometry.AxisAlignedBoundingBox(
                        np.minimum(self.ctrl_clicks[0], self.ctrl_clicks[1]),
                        np.maximum(self.ctrl_clicks[0], self.ctrl_clicks[1]))
                    self._add_box_geometry()
                    self.show_btn.enabled = True
                    self.export_btn.enabled = True
                self.window.set_needs_layout()

            self.widget.scene.scene.render_to_depth_image(depth_cb)
            return gui.Widget.EventCallbackResult.CONSUMED
        return gui.Widget.EventCallbackResult.IGNORED

    def show_cropped(self):
        print("[DEBUG] Show Cropped Cloud")
        cropped = self.pcd.crop(self.aabb)
        if self.widget.scene.has_geometry("cropped"):
            self.widget.scene.remove_geometry("cropped")
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = (*self.point_color, 1.0)
        mat.point_size = self.point_size
        self.widget.scene.add_geometry("cropped", cropped, mat)

    def export_cropped(self):
        print("[DEBUG] Export Cropped Cloud")
        cropped = self.pcd.crop(self.aabb)
        o3d.io.write_point_cloud("cropped.pcd", cropped)
        print(f"[DEBUG] Saved {len(cropped.points)} points")

    # --- UI 콜백들 ---
    def on_size_changed(self, value):
        self.point_size = int(value)
        print(f"[DEBUG] point_size -> {self.point_size}")
        self._add_pcd_geometry()
        self._update_cropped_geometry()

    def on_point_color_changed(self, color: gui.Color):
        # color.red, color.green, color.blue 속성으로 값을 읽어올 수 있습니다.
        self.point_color = [color.red, color.green, color.blue]
        self._add_pcd_geometry()

    def on_box_color_changed(self, color: gui.Color):
        self.box_color = [color.red, color.green, color.blue]
        if hasattr(self, "aabb"):
            self._add_box_geometry()

def main():
    # .ply 파일 절대 경로로 바꿔주세요
    app = BoundingBoxApp("/home/myungw00/ROS/gm/Code/data/scripts/0424-0515/2025-04-22/os1/2025-04-22_17-19-59.pcd")
    gui.Application.instance.run()

if __name__ == "__main__":
    main()

