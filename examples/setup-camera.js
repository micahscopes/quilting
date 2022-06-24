import interactionEvents from 'normalized-interaction-events';

export default (element, camera, radiansPerHalfScreenWidth = Math.PI/4) => {
    return interactionEvents(element)
        .on('wheel', function (ev) {
            camera.zoom(ev.x, ev.y, Math.exp(-ev.dy) - 1.0);
            ev.originalEvent.preventDefault();
        })
        .on('mousemove', function (ev) {
            if (!ev.active || ev.buttons !== 1) return;

            if (ev.mods.shift) {
                camera.pan(ev.dx, ev.dy);
            } else if (ev.mods.meta) {
                camera.pivot(ev.dx, ev.dy);
            } else {
                console.log('rotating...', ev)
                camera.rotate(
                    -ev.dx * radiansPerHalfScreenWidth,
                    -ev.dy * radiansPerHalfScreenWidth
                );
            }
            
            console.log(ev)
            ev.originalEvent.preventDefault();
        })
        .on('touchmove', function (ev) {
            if (!ev.active) return;
            camera.rotate(
                -ev.dx * radiansPerHalfScreenWidth,
                -ev.dy * radiansPerHalfScreenWidth
            );
            ev.originalEvent.preventDefault();
        })
        .on('pinchmove', function (ev) {
            if (!ev.active) return;
            camera.zoom(ev.x, ev.y, 1 - ev.zoomx);
            camera.pan(ev.dx, ev.dy);
        })
        .on('touchstart', ev => ev.originalEvent.preventDefault())
        .on('pinchstart', ev => ev.originalEvent.preventDefault())
}