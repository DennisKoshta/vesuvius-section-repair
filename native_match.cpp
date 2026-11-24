#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

// Propagate a labeled papyrus profile through native M7 values on a regular chart.
// Inputs are indexed axial z, arc length, native radius. Output radius uses the
// field's local radial index. No target reference mesh enters these operations.
struct Match {
    float radius;
    float quality;
};

struct Matcher {
    int half, search, depth;
    bool snap;
    std::vector<float> weights;

    Matcher(int h, int s, int d, bool center) : half(h), search(s), depth(d), snap(center), weights(2*h+1) {
        float mass=0;
        for (int n=-h;n<=h;++n) {
            float t=n/(.65f*h);
            mass+=(weights[n+h]=std::exp(-.5f*t*t));
        }
        for (auto &w:weights) w/=mass;
    }

    Match edge(const uint8_t *before,const uint8_t *after,float at,float recent) const {
        int rad=std::clamp((int)std::lround(at),half+search+3,depth-half-search-4);
        float mean_a=0,var_a=0;
        for (int h=-half;h<=half;++h) {
            float w=weights[h+half],a=before[rad+h];
            mean_a+=w*a;
            var_a+=w*a*a;
        }
        var_a=std::max(0.f,var_a-mean_a*mean_a);
        if (var_a<4) return {at,0.f};
        std::array<float,33> scores{}, correlations{};
        int win=0;
        for (int k=-search;k<=search;++k) {
            float mean_b=0,var_b=0,cov=0;
            for (int h=-half;h<=half;++h) {
                const float w=weights[h+half],a=before[rad+h],b=after[rad+h+k];
                mean_b+=w*b;
                var_b+=w*b*b;
                cov+=w*a*b;
            }
            var_b=std::max(0.f,var_b-mean_b*mean_b);
            float corr=(cov-mean_a*mean_b)/std::max(2.f,std::sqrt(var_a*var_b));
            correlations[k+search]=corr;
            scores[k+search]=corr-.004f*std::abs(k)-.003f*std::abs(k-std::clamp(recent,-4.f,4.f));
            if (scores[k+search]>scores[win]) win=k+search;
        }
        float shift=win-search;
        if (win>0 && win<2*search) {
            float den=scores[win-1]-2*scores[win]+scores[win+1];
            if (den<-.0001f)
                shift+=std::clamp(.5f*(scores[win-1]-scores[win+1])/den,-.5f,.5f);
        }
        float r=at+shift,quality=correlations[win];
        if (quality<.18f) r=at;
        if (snap && quality>.25f) {
            int ctr=std::clamp((int)std::lround(r),4,depth-5);
            int peak=ctr;
            for (int j=ctr-3;j<=ctr+3;++j)
                if (after[j]-.6f*std::abs(j-r)>after[peak]-.6f*std::abs(peak-r)) peak=j;
            if (after[peak]>40) {
                float sum=0,pos=0;
                for (int j=peak-2;j<=peak+2;++j) {
                    float m=std::max(0.f,(float)after[j]-.45f*after[peak]);
                    sum+=m; pos+=m*j;
                }
                r=.15f*r+.85f*(sum>0?pos/sum:peak);
            }
        }
        return {std::clamp(r,(float)(half+search+2),(float)(depth-half-search-3)),quality};
    }
};

extern "C" void transport_brackets(const uint8_t* field, int height,int width,int depth,
    const float* anchors,int gap,int half,int search,float *forward,float *backward,float *quality) {
    Matcher matcher(half,search,depth,false);
    #pragma omp parallel for schedule(static)
    for(int j=0;j<width;++j) {
        for(int top=0;top+gap<height;top+=gap) {
            int bottom=top+gap;
            int a=top*width+j,b=bottom*width+j;
            if(!std::isfinite(anchors[a])||!std::isfinite(anchors[b])) continue;
            const float margin=half+search+2;
            if(anchors[a]<margin||anchors[a]>=depth-margin||anchors[b]<margin||anchors[b]>=depth-margin)continue;
            float r=anchors[a],prev=0,acc=0;
            forward[a]=r;
            for(int i=top+1;i<=bottom;++i) {
                int k=i*width+j;
                auto q=matcher.edge(field+long(k-width)*depth,field+long(k)*depth,r,prev);
                prev=q.radius-r;r=q.radius;forward[k]=r;acc+=q.quality;
                quality[k]=acc/(i-top);
            }
            r=anchors[b];prev=0;acc=0;backward[b]=r;
            for(int i=bottom-1;i>=top;--i) {
                int k=i*width+j;
                auto q=matcher.edge(field+long(k+width)*depth,field+long(k)*depth,r,prev);
                prev=q.radius-r;r=q.radius;backward[k]=r;acc+=q.quality;
                quality[k]=std::min(quality[k],acc/(bottom-i));
            }
        }
    }
}
