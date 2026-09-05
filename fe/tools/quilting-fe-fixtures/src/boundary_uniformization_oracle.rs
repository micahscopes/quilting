//! Independent f64 replay, not a production sampler. Separates the numerical
//! boundary approximation from the quality of its interior extension.
use crate::clifford_oracle::weighted_paper_sample;

const FIXTURES: [(&str, [f64; 4]); 4] = [
    ("neutral", [1.0; 4]),
    ("squeezed", [0.02, 0.4, 0.4, 8.0]),
    ("reverse-squeezed", [8.0, 0.4, 0.4, 0.02]),
    ("unequal-axes", [0.02, 0.04, 4.0, 8.0]),
];

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt()
}
fn curve(edge: usize, t: f64) -> [f64; 2] {
    match edge { 0 => [t,0.0], 1 => [1.0,t], 2 => [t,1.0], 3 => [0.0,t], _ => [t,t] }
}
fn bias(t: f64, k: f64) -> f64 {
    if t == 0.0 || t == 1.0 || k == 1.0 { t } else { k*t / (1.0+(k-1.0)*t) }
}

struct EdgeTable { cumulative: Vec<f64>, bias: f64 }
impl EdgeTable {
    fn build(w: [f64;4], edge: usize, segments: usize) -> Self {
        let k=match edge {0=>w[0]/w[1],1=>w[1]/w[3],2=>w[2]/w[3],3=>w[0]/w[2],_ =>(w[0]/w[3]).sqrt()};
        let mut cumulative=vec![0.0];
        let mut previous=weighted_paper_sample(w,curve(edge,0.0));
        for i in 1..=segments {
            let p=weighted_paper_sample(w,curve(edge,bias(i as f64/segments as f64,k)));
            cumulative.push(cumulative.last().unwrap()+distance(previous,p));
            previous=p;
        }
        Self {cumulative,bias:k}
    }
    fn length(&self) -> f64 { *self.cumulative.last().unwrap() }
    fn parameter(&self, fraction: f64) -> f64 {
        if fraction==0.0 || fraction==1.0 {return fraction;}
        let target=fraction*self.length();
        let high=self.cumulative.partition_point(|&s|s<target).clamp(1,self.cumulative.len()-1);
        let low=high-1;
        let f=(target-self.cumulative[low])/(self.cumulative[high]-self.cumulative[low]);
        bias((low as f64+f)/(self.cumulative.len()-1) as f64,self.bias)
    }
    fn length_at(&self, parameter: f64) -> f64 {
        let coordinate=bias(parameter,1.0/self.bias)*(self.cumulative.len()-1) as f64;
        let low=(coordinate.floor() as usize).min(self.cumulative.len()-2);
        let f=(coordinate-low as f64).clamp(0.0,1.0);
        self.cumulative[low]*(1.0-f)+self.cumulative[low+1]*f
    }
    fn lod(&self) -> u32 {
        (0..=7).find(|&n|0.18*(1_u32<<n) as f64>=self.length()).unwrap_or(7)
    }
}

#[test]
fn boundary_tables_uniformize_curved_lengths_in_independent_f64_replay() {
    for (name,w) in FIXTURES {
        for edge in 0..5 {
            let table=EdgeTable::build(w,edge,1024);
            let coarse=EdgeTable::build(w,edge,512);
            let reference=EdgeTable::build(w,edge,16384);
            let segments=1_u32<<table.lod();
            let target=reference.length()/segments as f64;
            let mut previous=0.0;
            let mut worst=0.0_f64;
            let mut raw_worst=0.0_f64;
            let mut raw_previous=0.0;
            for i in 1..=segments {
                let fraction=i as f64/segments as f64;
                let length=reference.length_at(table.parameter(fraction));
                worst=worst.max(((length-previous)/target-1.0).abs());
                previous=length;
                let raw_length=reference.length_at(fraction);
                raw_worst=raw_worst.max(((raw_length-raw_previous)/target-1.0).abs());
                raw_previous=raw_length;
            }
            assert!(worst<0.005,"{name} edge {edge}: local spacing error {worst} exceeds 0.5% replay gate");
            eprintln!("boundary {name} edge={edge}: lod={} length={:.9} coarse/fine={:.3e} local_spacing_rel={worst:.3e} native_spacing_rel={raw_worst:.3e}",
                table.lod(),table.length(),(table.length()-coarse.length()).abs()/table.length());
        }
    }
}

fn edge_slice(p: [f64;3], table: &EdgeTable, reversed: bool) -> [f64;3] {
    if p[1]==0.0 || p[2]==0.0 {return p;}
    let span=p[1]+p[2];
    let t=p[2]/span;
    let mapped=if reversed {1.0-table.parameter(1.0-t)} else {table.parameter(t)};
    if mapped==t {return p;}
    let c=span*((1.0-span)*t+span*mapped);
    [p[0],span-c,c]
}
fn half_parameter(p: [f64;3], half:usize, tables:&[EdgeTable;5], warped:bool) -> [f64;2] {
    let edge_map=|edge:usize,t| if warped {tables[edge].parameter(t)} else {t};
    let a=p[0]; let b=p[1]; let c=p[2];
    if half==0 {
        if a==0.0 {return curve(1,edge_map(1,c));}
        if b==0.0 {return curve(4,edge_map(4,c));}
        if c==0.0 {return curve(0,edge_map(0,b));}
    } else {
        if a==0.0 {return curve(2,edge_map(2,b));}
        if b==0.0 {return curve(3,edge_map(3,c));}
        if c==0.0 {return curve(4,edge_map(4,b));}
    }
    let mut q=p;
    if warped {
        let laws=if half==0 {[(1,false),(4,true),(0,false)]} else {[(2,true),(3,true),(4,false)]};
        for (edge,reversed) in laws {
            let r=edge_slice(q,&tables[edge],reversed);
            q=[r[1],r[2],r[0]];
        }
    }
    if half==0 {[q[1]+q[2],q[2]]} else {[q[1],q[1]+q[2]]}
}
fn signed_area(uv:[[f64;2];3]) -> f64 {
    0.5*((uv[1][0]-uv[0][0])*(uv[2][1]-uv[0][1])-(uv[1][1]-uv[0][1])*(uv[2][0]-uv[0][0]))
}

#[test]
fn uniform_boundaries_do_not_certify_interior_atlas_quality() {
    let path=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../ingots/demos/classic_quilting_lod/assets/sha256/695152acd242f5b88a6ac4074f6855fea35b12bf16436308ba925b8cc962803e.bin");
    let atlas=crate::decode(&std::fs::read(path).unwrap()).unwrap();
    for (name,w) in FIXTURES {
        let tables:[EdgeTable;5]=std::array::from_fn(|i|EdgeTable::build(w,i,1024));
        for step in 0..=256 {
            let t=step as f64/256.0;
            assert_eq!(half_parameter([1.0-t,0.0,t],0,&tables,true),half_parameter([1.0-t,t,0.0],1,&tables,true),
                "one canonical diagonal parameter sequence");
        }
        let diagonal: [Vec<[f64;2]>;2]=std::array::from_fn(|half| {
            let edges=if half==0 {[1,4,0]} else {[2,3,4]};
            let requested=edges.map(|i|1_u32<<tables[i].lod());
            let mut permutation=[0,1,2];
            permutation.sort_by_key(|&i|requested[i]);
            let canonical=permutation.map(|i|requested[i]);
            let tile=atlas.patches.iter().find(|p|p.key==crate::AtlasKey::new(canonical[0],canonical[1],canonical[2])).unwrap();
            let mut points=Vec::new();
            for tri in &atlas.triangles[tile.first_triangle as usize..(tile.first_triangle+tile.triangle_count) as usize] {
                for id in tri.indices {
                    let raw=atlas.vertices[id as usize].barycentric.map(f64::from);
                    let mut p=[0.0;3];
                    for i in 0..3 {p[permutation[i]]=raw[i];}
                    if p[if half==0 {1}else{2}]==0.0 {points.push(half_parameter(p,half,&tables,true));}
                }
            }
            points.sort_by(|a,b|a[0].total_cmp(&b[0]));
            points.dedup();
            points
        });
        assert_eq!(diagonal[0].len(),(1_usize<<tables[4].lod())+1,"all requested diagonal vertices resident");
        assert_eq!(diagonal[0],diagonal[1],"actual permuted atlas tiles agree on the full diagonal sequence");
        let mut counts=[0;2];
        for (mode,warped) in [false,true].into_iter().enumerate() {
            let mut chord_error=0.0_f64;
            let mut minimum_quality=1.0_f64;
            let mut inverted=0;
            let mut area_sum=0.0;
            let mut absolute_area=0.0;
            for half in 0..2 {
                let edges=if half==0 {[1,4,0]} else {[2,3,4]};
                let requested=edges.map(|i|1_u32<<tables[i].lod());
                let mut permutation=[0,1,2];
                permutation.sort_by_key(|&i|requested[i]);
                let canonical=permutation.map(|i|requested[i]);
                let tile=atlas.patches.iter().find(|p|p.key==crate::AtlasKey::new(canonical[0],canonical[1],canonical[2])).unwrap();
                for tri in &atlas.triangles[tile.first_triangle as usize..(tile.first_triangle+tile.triangle_count) as usize] {
                    let mut bary=tri.indices.map(|id| {
                        let raw=atlas.vertices[id as usize].barycentric.map(f64::from);
                        let mut p=[0.0;3];
                        for i in 0..3 {p[permutation[i]]=raw[i];}
                        p
                    });
                    if signed_area(bary.map(|p|half_parameter(p,half,&tables,false)))<0.0 {bary.swap(1,2);}
                    let uv=bary.map(|p|half_parameter(p,half,&tables,warped));
                    let area=signed_area(uv);
                    inverted+=usize::from(area<=0.0);
                    area_sum+=area;
                    absolute_area+=area.abs();
                    let points=uv.map(|p|weighted_paper_sample(w,p));
                    let lengths=[distance(points[0],points[1]),distance(points[1],points[2]),distance(points[2],points[0])];
                    let semiperimeter=lengths.iter().sum::<f64>()*0.5;
                    let area3=(semiperimeter*(semiperimeter-lengths[0])*(semiperimeter-lengths[1])*(semiperimeter-lengths[2])).max(0.0).sqrt();
                    minimum_quality=minimum_quality.min(4.0*3.0_f64.sqrt()*area3/lengths.iter().map(|x|x*x).sum::<f64>());
                    for bary in [[1.0/3.0;3],[0.5,0.5,0.0],[0.0,0.5,0.5],[0.5,0.0,0.5]] {
                        let parameter=std::array::from_fn(|i|(0..3).map(|j|bary[j]*uv[j][i]).sum());
                        let flat=std::array::from_fn(|i|(0..3).map(|j|bary[j]*points[j][i]).sum());
                        chord_error=chord_error.max(distance(weighted_paper_sample(w,parameter),flat));
                    }
                    counts[mode]+=1;
                }
            }
            assert!((area_sum-1.0).abs()<0.00001,"signed parameter domain coverage");
            assert!(chord_error.is_finite() && minimum_quality.is_finite());
            eprintln!("interior {name} warped={warped}: triangles={} sampled_max_chord={chord_error:.7} min_quality={minimum_quality:.7} inverted={inverted} absolute_area={absolute_area:.9}",counts[mode]);
        }
        assert_eq!(counts[0],counts[1],"compare the same atlas triangle budget");
    }
}
