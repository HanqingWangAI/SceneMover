#include <iostream>
#include <queue>
#include <cstring>
// #include <bits/stdc++.h>
#include <map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <thread>
#include <pthread.h>
using namespace std;



const int maxn = 300;
const float pi = 3.141592653589793238;
const float EPS = 1e-3;
int dis[maxn][maxn];
int sum[maxn][maxn];
int dist[maxn][maxn][50];

int _x[4] = {-1,1,0,0}, _y[4] = {0,0,-1,1};

struct node{
    int f,d,state;
    float x,y;
    node(){}
    node(int f,int d, float x, float y,int state=0):f(f),d(d),x(x),y(y),state(state){}
    bool operator < (const node &a) const{
        return f > a.f;
    }

};

struct point{
    float x, y;
    int state;
    point(){}
    point(float x, float y, int state=0):x(x),y(y),state(state){}
    bool operator<(const point &a) const {
        if(int(x) == int(a.x)){
            if(int(y) == int(a.y))
                return state < a.state;
            return y < a.y;
        }
        return x < a.x;
    }
    bool operator==(const point &a) const{
        return int(x) == int(a.x) && int(y) == int(a.y) && state == a.state;
    }

    point operator+(const point &a){
        return move(point(x+a.x, y+a.y));
    }

    point operator-(const point &a){
        return move(point(x-a.x, y-a.y));
    }

    point operator*(const point &a){
        return move(point(x*a.x, y*a.y));
    }

    point operator*(const float a){
        return move(point(x*a, y*a));
    }

    point operator/(const float a){
        return move(point(x/a, y/a));
    }

    

    float operator&(const point &a){
        return x * a.y - y * a.x;
    }

    float length() const {
        return sqrt(pow(x, 2.0)+pow(y, 2.0));
    }

    float dot(const point &a) const {
        return x * a.x + y * a.y;
    }

    float cmp(const point &a) const {
        return fabs(x-a.x) + fabs(y-a.y);
    }

};

point operator*(const float a, const point &b){
    return point(a*b.x, a*b.y);
}

bool compare(const point &a, const point &b){
    if(a.x == b.x){
        return a.y < b.y;
    }
    return a.x < b.x;
}

pair<float,float> cross_point(point a1, point a2, point b1, point b2){
    float x2_x2, y2_y2, x1x2, y1y2, y1_y2_, x1_x2_;
    x2_x2 = b2.x - a2.x;
    y2_y2 = b2.y - a2.y;
    x1x2 = a1.x - a2.x;
    y1y2 = a1.y - a2.y;
    y1_y2_ = b1.y - b2.y;
    x1_x2_ = b1.x - b2.x;
    
    float beta,alpha;
    if(fabs(y1_y2_ * x1x2 - x1_x2_*y1y2) < EPS){
        beta = 10000000;
    }
    else{
        beta = (x2_x2*y1y2 - y2_y2*x1x2)/(y1_y2_ * x1x2 - x1_x2_*y1y2);
    }

    if(x1x2 == 0){
        alpha = (y2_y2 + y1_y2_ * beta)/y1y2;
    }
    else{
        alpha = (x2_x2+x1_x2_*beta)/x1x2;
    }

    return move(make_pair(alpha, beta));

}

float polygon_area(vector<point> const &points){
    float area = 0;
    for(int i = 0;i < points.size();i++){
        point a = points[i];
        point b = points[(i+1)%points.size()];
        // printf("a (%f,%f) b (%f,%f) %f\n",a.x,a.y,b.x,b.y,a&b);
        area += a&b;
    }
    return 0.5*area;
}

vector<point> unique(vector<point> ar){
    vector<point> ps;
    int n = ar.size();
    if(n > 0){
        for(int i = 0; i < n-1; i++){
            if(ar[i].cmp(ar[(i+1)%n])>EPS){
                ps.push_back(ar[i]);
            }
        }
        ps.push_back(ar[n-1]);
    }
    return ps;
}

float cos(point const &a, point const &b){
    return a.dot(b)/(a.length()*b.length()+EPS);
}

point start;

bool cmp(point &a, point &b){
    return cos(a-start, point(0,1)) < cos(b-start, point(0,1));
}

void clock_wise(vector<point> &ps){
    if(ps.size() == 0)
        return ;
    
    sort(ps.begin(),ps.end(),compare);// first sort the points by their cords
    start = ps[0];
    
    sort(ps.begin()+1,ps.end(),cmp); // sort the points by their cos value
    reverse(ps.begin(),ps.end());
}

void cross_detect(point const &a1, point const &a2, vector<point> const &pb, bool *flag){
    int nb = pb.size();
    for(int i = 0;i < nb;i++){
        point b1 = pb[i];
        point b2 = pb[(i+1)%nb];
        pair<float,float> pair;
        pair = cross_point(a1,a2,b1,b2);
        float a, b;
        a = pair.first;
        b = pair.second;
        if(a >= 0 && a <= 1 && b >= 0 && b <= 1){
            *flag = true;
            break;
        }
    }
}

bool convex_cross(vector<point> const &pa, vector<point> const &pb){
    int na,nb;
    na = pa.size();
    nb = pb.size();
    // vector<thread> threads;
    // bool flags[na];
    // int cnt = 0;

    for(int i = 0; i < na; i++){
        // point a1 = pa[i];
        // point a2 = pa[(i+1)%na];
        // threads.emplace_back(thread(cross_detect,a1,a2,pb,flags+i));
        for(int j = 0;j < nb;j++){
            pair<float,float> p = cross_point(pa[i],pa[(i+1)%na],pb[j],pb[(j+1)%nb]);
            float a, b;
            a = p.first;
            b = p.second;
            if(a >= 0 && a <= 1 && b >= 0 && b <= 1){
                return true;
            }
        }
        // for(int j = 0;j < nb; j++){
        //     point b1 = pb[j];
        //     point b2 = pb[(j+1)%nb];
            
        //     // thread th(cross_detect,a1,a2,b1,b2,flags+cnt);
        //     threads.emplace_back(thread(cross_detect,a1,a2,b1,b2,flags+cnt));
        //     cnt ++;
        //     // pair<float,float> pair;
        //     // pair = cross_point(a1,a2,b1,b2);
        //     // float a, b;
        //     // a = pair.first;
        //     // b = pair.second;
        //     // if(a >= 0 && a <= 1 && b >= 0 && b <= 1){
        //     //     return true;
        //     // }
        // }
    }
    
    // for(int i = 0; i < threads.size();i++){
    //     threads[i].join();
    //     // if(flags[i]) return true;
    // }
    // for(int i = 0;i < threads.size();i++){
    //     if(flags[i]) return true;
    // }
    return false;
}

float CPIA(vector<point> const &pa_, vector<point> const &pb_){
    int na,nb;
    na = pa_.size();
    nb = pb_.size();
    vector<point> pa,pb;
    pa = pa_;
    pb = pb_;
    clock_wise(pa);
    clock_wise(pb);
    vector<point> ps;
    // printf("pa =====================\n");
    // for(int i = 0; i < na; i++){
    //     printf("%f %f\n",pa[i].x,pa[i].y);
    // }
    // printf("pb ===============------\n");
    // for(int i = 0;i < nb; i++){
    //     printf("%f %f\n",pb[i].x,pb[i].y);
    // }
    for(int i = 0; i < na; i++){
        point a1 = pa[i];
        point a2 = pa[(i+1)%na];
        bool flag = true;
        int last;
        // printf("a1 %d (%f,%f)\n",i,a1.x,a1.y);
        for(int j = 0; j < nb; j++){
            point b1 = pb[j];
            point b2 = pb[(j+1)%nb];
            // printf("b1->b2 (%f,%f) b1->a1 (%f,%f) cross_product:%f\n",(b2-b1).x,(b2-b1).y,(a1-b1).x,(a1-b1).y,((b2-b1) & (a1-b1)));
            int now;
            if(((b2-b1) & (a1-b1)) > EPS){
                now = -1;
                // break;
            }
            else{
                now = 1;
            }
            if(j != 0){
                if(last != now){
                    flag = false;
                }
            }
            last = now;
        }

        if(flag){
            ps.push_back(a1);
            // printf("push %d\n",i);
        }

        for(int j = 0;j < nb; j++){
            point b1 = pb[j];
            point b2 = pb[(j+1)%nb];
            pair<float,float> pair;
            pair = cross_point(a1,a2,b1,b2);
            float a, b;
            a = pair.first;
            b = pair.second;
            if(a >= 0 && a <= 1 && b >= 0 && b <= 1){
                point p;
                p = a * a1 + (1-a) * a2;
                // printf("cross %f %f\n",a,b);
                // printf("a1(%f,%f),a2(%f,%f),b1(%f,%f),b2(%f,%f)\n",a1.x,a1.y,a2.x,a2.y,b1.x,b1.y,b2.x,b2.y);
                ps.push_back(p);
            }
        }
    }
    // printf("========================------------------==========\n");
    
    for(int i = 0; i < nb; i++){
        point a1;
        a1 = pb[i];
        bool flag = true;
        int last;
        // printf("a1 %d (%f,%f)\n",i,a1.x,a1.y);
        for(int j = 0;j < na; j++){
            point b1,b2;
            b1 = pa[j];
            b2 = pa[(j+1)%na];
            int now;
            // printf("b1->b2 (%f,%f) b1->a1 (%f,%f) cross_product:%f\n",(b2-b1).x,(b2-b1).y,(a1-b1).x,(a1-b1).y,((b2-b1) & (a1-b1)));
            if(((b2-b1)&(a1-b1)) > EPS){
                // flag = false;
                now = -1;
                // break;
            }
            else{
                now = 1;
            }
            if(j != 0){
                if(last != now){
                    flag = false;
                }
            }
            last = now;
        }

        if(flag){
            // printf("push %d\n",i);
            ps.push_back(a1);
        }
    }
    // printf("%d\n",ps.size());
    sort(ps.begin(),ps.end(),compare);
    ps = unique(ps);
    
    // printf("OK\n");
    // printf("%d\n",ps.size());
    clock_wise(ps);
    // if(ps.size()>0){
    //     printf("ps ===============\n");
    //     for(int i = 0;i < ps.size(); i++){
    //         printf("%f %f\n",ps[i].x,ps[i].y);
    //     }
    //     printf("area %f\n",polygon_area(ps));
    // }
    return polygon_area(ps);

}

float NPIA(vector<point> const &pa, vector<point> const &pb){
    int na, nb;
    na = pa.size();
    nb = pb.size();
    float res = 0;
    for(int i = 1; i < na-1; i++){
        vector<point> sa;
        sa.push_back(pa[0]);
        sa.push_back(pa[i]);
        sa.push_back(pa[i+1]);
        float sign = 1;
        if(polygon_area(sa) < 0){
            sign = -1;
        }
        for(int j = 1;j < nb-1; j++){
            vector<point> sb;
            sb.push_back(pb[0]);
            sb.push_back(pb[j]);
            sb.push_back(pb[j+1]);
            // printf("step in\n");
            float sign2 = 1;
            if(polygon_area(sb) < 0){
                sign2 = -1;
            }
            // if(fabs(CPIA(sa,sb))>EPS){
            //     printf("sa:\n");
            //     for(int i = 0;i < sa.size();i++){
            //         printf("%f %f\n",sa[i].x,sa[i].y);
            //     }

            //     printf("sb:\n");
            //     for(int i = 0;i < sb.size();i++){
            //         printf("%f %f\n",sb[i].x,sb[i].y);
            //     }
            //     printf("value %f\n",fabs(CPIA(sa,sb)));
            //     printf("============================\n");
            // }
            res += sign*sign2*fabs(CPIA(sa,sb));
            // printf("step out\n");
        }
    }
    return fabs(res);
}




void rotate(float *opoints, float *points, int n, float radian, float &blx, float &bly, float &brx, float &bry){
    if(n == 0){
        blx = bly = brx = bry = 0;
        return;
    }
    float lx,ly,rx,ry,mx,my;
    lx = ly = 10000;
    rx = ry = -10000;
    float eps = 1e-6;
    // printf("size %d\n",sizeof(opoints));
    for(int i = 0;i < n;i++){
        points[i*2] = opoints[i*2];
        points[i*2+1] = opoints[i*2+1];
    }

    for(int i = 0;i < n;i++){
        lx = min(lx, points[i<<1]);
        rx = max(rx, points[i<<1]);
        ly = min(ly, points[i<<1|1]);
        ry = max(ry, points[i<<1|1]);
    }
    mx = 0.5*(lx+rx);
    my = 0.5*(ly+ry);
    

    for(int i = 0; i < n; i++){
        points[i<<1] -= mx;
        points[i<<1|1] -= my;
    }
    mx -= lx;
    my -= ly;

    float rot[2][2];
    rot[0][0] = cos(radian);
    rot[0][1] = -sin(radian);
    rot[1][0] = sin(radian);
    rot[1][1] = cos(radian);

    float res[n][2];
    memset(res,0,sizeof(res));
    for(int i = 0; i < 2; i++){
        for(int j = 0;j < n;j++){
            for(int k = 0;k < 2;k++){
                res[j][i] += rot[i][k] * points[j<<1|k]; 
            }
        }
    }

    blx = bly = 10000;
    brx = bry = -10000;
    for(int i = 0;i < n;i++){
        // points[i<<1] = round(res[i][0] + mx - 0.5 + eps);
        // points[i<<1|1] = round(res[i][1] + my - 0.5 + eps);
        points[i<<1] = res[i][0] + eps;
        points[i<<1|1] = res[i][1] + eps;
        blx = min(blx, points[i<<1]);
        brx = max(brx, points[i<<1]);
        bly = min(bly, points[i<<1|1]);
        bry = max(bry, points[i<<1|1]);
    }
    // if(blx > 0 && bly > 0){
    //     printf("error %f %f %f %f %d\n",blx,bly,brx,bry,n);
    // }
}



bool check(int *mat, float *opoints, int num, float radian, float x, float y, int n, int m){
    float points[num*2];
    float xmax,xmin,ymax,ymin;
    rotate(opoints,points,num,radian,xmin,ymin,xmax,ymax);
    // x = max(0.0, x - 0.5 * (xmax - xmin));
    // y = max(0.0, y - 0.5 * (ymax - ymin));

    if(x+xmax < n && x+xmin >= 0 && y+ymax < m && y+ymin >= 0){
        // printf("enter\n");
        int aa,bb,ab,ba;
        aa = sum[int(x+xmax)][int(y+ymax)];
        ab = ba = bb = 0;

        if(int(x+xmin) != 0)
            ba = sum[int(x+xmin-1)][int(y+ymax)];
        
        if(int(y+ymin) != 0)
            ab = sum[int(x+xmax)][int(y+ymin-1)];
        
        if(int(x+xmin) != 0 && int(y+ymin) != 0)
            bb = sum[int(x+xmin-1)][int(y+ymin-1)];
        
        int res = aa-ab-ba+bb;
        bool push_flag = false;
        // res = 1;
        if(res == 0){
            push_flag = true;
        }
        else{
            push_flag = true;
            for(int i = 0;i < num;i++){
                int __x,__y;
                __x = x+points[i<<1];
                __y = y+points[i<<1|1];
                if(mat[__x*m+__y] != 0){
                    push_flag=false;
                    break;
                }
            }
        }
        // delete points;
        return push_flag;
    }
    // delete points;
    return false;

}

bool check_poly(int *mat, float *shapes, float *opoints, int *amounts, int *pn, int total, int id, float radian, float x, float y, int n, int m){
    int num = pn[id];
    float points[num*2];
    float xmax,xmin,ymax,ymin;
    rotate(opoints,points,num,radian,xmin,ymin,xmax,ymax);
    vector<point> va[total],ps;

    if(x+xmax < n && x+xmin >= 0 && y+ymax < m && y+ymin >= 0){
        // printf("enter\n");
        int aa,bb,ab,ba;
        aa = sum[int(x+xmax)][int(y+ymax)];
        ab = ba = bb = 0;

        if(int(x+xmin) != 0)
            ba = sum[int(x+xmin-1)][int(y+ymax)];
        
        if(int(y+ymin) != 0)
            ab = sum[int(x+xmax)][int(y+ymin-1)];
        
        if(int(x+xmin) != 0 && int(y+ymin) != 0)
            bb = sum[int(x+xmin-1)][int(y+ymin-1)];
        
        int res = aa-ab-ba+bb;
        
        if(res == 0)
            return true;

    }
    else{
        return false;
    }
    
    xmax += x;
    xmin += x;
    ymax += y;
    ymin += y;
    
    for(int i = 0;i < pn[id];i++){
        float xx,yy;
        xx = points[i<<1]+x;
        yy = points[i<<1|1]+y;

        if(xx < 0 || xx > n || yy < 0 || yy > m)
            return false;

        ps.push_back(point(xx,yy));
    }

    // for(int i = 0;i < ps.size();i++){
    //     point pa,pb;
    //     pa = ps[i];
    //     pb = ps[(i+1)%ps.size()];
    //     int loop = max(fabs(pb.x - pa.x) + EPS + 1,fabs(pb.y-pa.y) + EPS + 1);
    //     for(int j = 0;j < loop+1;j++){
    //         point pt = (1.0-1.0*j/loop)*pa+(1.0*j/loop)*pb;
    //         int x,y;
    //         x = pt.x + EPS;
    //         y = pt.y + EPS;
    //         // printf("%d %d\n",x,y);
    //         if(mat[x*m+y] == 1){
    //             // printf("collide\n");
    //             return false;
    //         }
    //     }
    // }
    vector<int> ss;
    for(int i = 0;i < total; i++){
        if(i == id) continue;
        num = pn[i];
        float x1,x2,y1,y2;
        x1 = y1 = 100000;
        x2 = y2 = -100000;
        for(int j = 0;j < num; j++){
            float x_,y_;
            x_ = shapes[(amounts[i]+j)<<1];
            y_ = shapes[(amounts[i]+j)<<1|1];
            va[i].push_back(point(x_,y_));
            x1 = min(x1,x_);
            x2 = max(x2,x_);
            y1 = min(y1,y_);
            y2 = max(y2,y_);
        }
        x1 = max(x1,xmin);
        x2 = min(x2,xmax);
        y1 = max(y1,ymin);
        y2 = min(y2,ymax);
        if(x1 <= x2 && y1 <= y2)
            ss.push_back(i);
        
    }
    return false;

    for(int i = 0; i < ss.size(); i++){
        // if(i != id){
            // if(NPIA(ps,va[i]) > EPS){
            if(convex_cross(ps, va[ss[i]])){
                return false;
            }
        // }
    }
    
    return true;

}

bool check_poly_fast(int *mat, float *points, int num, float x, float y, int n, int m, float xmin, float xmax, float ymin, float ymax){
    // printf("xmin %f xmax %f ymin %f ymax %f\n",x + xmin,x + xmax,y+ymin,y+ymax);

    if(x+xmax < n && x+xmin >= 0 && y+ymax < m && y+ymin >= 0){
        // printf("enter\n");
        vector<point> ps;
        int aa,bb,ab,ba;
        aa = sum[int(x+xmax)][int(y+ymax)];
        ab = ba = bb = 0;

        if(int(x+xmin) != 0)
            ba = sum[int(x+xmin-1)][int(y+ymax)];
        
        if(int(y+ymin) != 0)
            ab = sum[int(x+xmax)][int(y+ymin-1)];
        
        if(int(x+xmin) != 0 && int(y+ymin) != 0)
            bb = sum[int(x+xmin-1)][int(y+ymin-1)];
        
        int res = aa-ab-ba+bb;
        
        if(res == 0)
            return true;

        for(int i = 0;i < num;i++){
            float xx, yy;
            xx = points[i<<1]+x;
            yy = points[i<<1|1]+y;

            if(xx < 0 || xx > n || yy < 0 || yy > m)
                return false;

            if(mat[int(xx+EPS)*m+int(yy+EPS)] != 0){
                // printf("%f %f\n",xx,yy);
                return false;
            }

            ps.emplace_back(point(xx,yy));
        }

        // for(int i = 0;i < ps.size();i++){
        //     point pa,pb;
        //     pa = ps[i];
        //     pb = ps[(i+1)%ps.size()];
        //     int loop = max(fabs(pb.x - pa.x) + EPS + 1,fabs(pb.y-pa.y) + EPS + 1);
        //     for(int j = 0;j < loop+1;j++){
        //         point pt = (1.0-1.0*j/loop)*pa+(1.0*j/loop)*pb;
        //         int x,y;
        //         x = pt.x + EPS;
        //         y = pt.y + EPS;
        //         // printf("%d %d\n",x,y);
        //         if(mat[x*m+y] != 0){
        //             // printf("%f %f\n",pt.x,pt.y);
        //             // printf("collide\n");
        //             return false;
        //         }
        //     }
        // }
        return true;
    }
    // printf("fuck\n");
    return false;
}

bool check_poly_fast_terrain(int *mat, int *terrain, float *points, float *prepoints, int num, float x, float y, int n, int m, float xmin, float xmax, float ymin, float ymax, int dir){
    // printf("xmin %f xmax %f ymin %f ymax %f\n",x + xmin,x + xmax,y+ymin,y+ymax);

    if(x+xmax < n && x+xmin >= 0 && y+ymax < m && y+ymin >= 0){
        // printf("enter\n");
        vector<point> ps,preps;
        int aa,bb,ab,ba;
        aa = sum[int(x+xmax)][int(y+ymax)];
        ab = ba = bb = 0;

        if(int(x+xmin) != 0)
            ba = sum[int(x+xmin-1)][int(y+ymax)];
        
        if(int(y+ymin) != 0)
            ab = sum[int(x+xmax)][int(y+ymin-1)];
        
        if(int(x+xmin) != 0 && int(y+ymin) != 0)
            bb = sum[int(x+xmin-1)][int(y+ymin-1)];
        
        int res = aa-ab-ba+bb;
        
        if(res == 0)
            return true;

        for(int i = 0;i < num;i++){
            float xx, yy, _xx, _yy;
            xx = points[i<<1]+x;
            yy = points[i<<1|1]+y;

            _xx = prepoints[i<<1]+x;
            _yy = prepoints[i<<1|1]+y;
            
            if(dir != -1){
                _xx -= _x[dir];
                _yy -= _y[dir];
            }

            if(xx < 0 || xx > n || yy < 0 || yy > m)
                return false;

            if(mat[int(xx+EPS)*m+int(yy+EPS)] != 0){
                // printf("%f %f\n",xx,yy);
                return false;
            }

            ps.emplace_back(point(xx,yy));
            preps.emplace_back(point(_xx,_yy));
        }

        for(int i = 0;i < ps.size();i++){
            point pa,pb,pa_,pb_;
            pa = ps[i];
            pb = ps[(i+1)%ps.size()];
            pa_ = preps[i];
            pb_ = preps[(i+1)%ps.size()];
            int loop = max(fabs(pb.x - pa.x) + EPS + 1,fabs(pb.y-pa.y) + EPS + 1);
            for(int j = 0;j < loop+1;j++){
                point pt = (1.0-1.0*j/loop)*pa+(1.0*j/loop)*pb;
                point pt_ = (1.0-1.0*j/loop)*pa_+(1.0*j/loop)*pb_;
                int x,y,dx,dy;
                x = pt.x + EPS;
                y = pt.y + EPS;
                dx = pt_.x + EPS;
                dy = pt_.y + EPS;
                // printf("%d %d\n",x,y);
                if(mat[x*m+y] != 0 && abs(terrain[x*m+y]-terrain[dx*m+dy]) < 10){
                    // printf("%f %f\n",pt.x,pt.y);
                    // printf("collide\n");
                    return false;
                }
            }
        }
        return true;
    }
    // printf("fuck\n");
    return false;
}

void init(){
    memset(dis,-1,sizeof(dis));
    memset(sum,0,sizeof(sum));
}



extern "C"{
    void search(int *mat, int n, int m,
            int id,
            int tx, int ty,
            int sx, int sy,
            int h, int w,
            int *lx, int *ly, int *len,
            int *ret
            )
    {

        init();
        priority_queue<node> pq;
        map<point, point> fa;



        for(int i = 0; i < h; i++){
            for(int j = 0; j < w; j++){
                mat[(sx+i)*m + (sy+j)] = 0;
            }
        }


        
        sum[0][0] = mat[0];

        for(int i = 1; i < n; i++)
            sum[i][0] = sum[i-1][0] + mat[i*m];
        
        for(int i = 1; i < m;i++)
            sum[0][i] = sum[0][i-1] + mat[i];
        
        for(int i = 1;i < n;i++)
            for(int j = 1; j < m;j++)
                sum[i][j] = sum[i][j-1] + sum[i-1][j] - sum[i-1][j-1] + mat[i*m+j];        
        
        int f = 0;
        bool l = true;
        for(int i = 0;i < h;i++){
            for(int j = 0;j < w;j++){
                if(mat[(sx+i)*m+(sy+j)] != 0){
                    l = false;
                    break;
                }
            }
            if(!l){
                break;
            }
        }
        if(l){
            f = 0 + abs(sx-tx) + abs(sy-ty);
            pq.push(node(f,0,sx,sy));
            dis[sx][sy] = f;
        }
        // printf("%d\n",pq.empty());
        while(!pq.empty()){
            node a = pq.top();
            // printf("x=%d y=%d d=%d\n",a.x,a.y,a.d);
            pq.pop();
            int x = a.x;
            int y = a.y;
            int d = a.d;
            if(x == tx && y == ty){
                break;
            }
            
            for(int i = 0;i < 4;i++){
                int tmpx,tmpy;
                tmpx = x + _x[i];
                tmpy = y + _y[i];
                if(tmpx + h <= n && tmpx >= 0 && tmpy + w <= m && tmpy >= 0){
                    int aa,bb,ab,ba;
                    aa = sum[tmpx+h-1][tmpy+w-1];
                    ab = ba = bb = 0;

                    if(tmpx != 0)
                        ba = sum[tmpx-1][tmpy+w-1];
                    
                    if(tmpy != 0)
                        ab = sum[tmpx+h-1][tmpy-1];
                    
                    if(tmpx != 0 && tmpy != 0)
                        bb = sum[tmpx-1][tmpy-1];
                    
                    int res = aa-ab-ba+bb;
                    // printf("res=%d\n",res);
                    if(res == 0){
                        int f_ = d + 1 + abs(tmpx-tx) + abs(tmpy-ty);
                        if(dis[tmpx][tmpy] == -1 || dis[tmpx][tmpy]>f_){
                            dis[tmpx][tmpy] = f_;
                            pq.push(node(f_,d+1,tmpx,tmpy));
                            fa[point(tmpx,tmpy)] = point(x,y);
                        }
                    }

                }
            }
            

        }

        for(int i = 0;i < h;i++){
            for(int j = 0;j < w;j++){
                mat[(sx+i)*m+(sy+j)] = id + 2;
            }
        }
        // for(int i = 0;i < n;i++){
        //     for(int j = 0;j < m;j++){
                
        //     }
        // }
        // printf("sx=%d sy=%d, tx=%d ty=%d\n",sx,sy,tx,ty);
        // for(int i =0;i < n;i++){
        //     for(int j =0;j < m;j++){
        //         printf("%d ",dis[i][j]);
        //     }
        //     printf("\n");
        // }
        // printf("============================================\n");
        bool flag = true;
        if(dis[tx][ty] == -1)
            flag = false;
        // printf("dis %d\n",dis[tx][ty]);
        if(flag){
            // printf("sx=%d sy=%d, tx=%d ty=%d\n",sx,sy,tx,ty);
            point cur_p = point(tx,ty);
            // point start = point(sx,sy);
            int cnt = 0;
            while(!(cur_p.x == sx && cur_p.y == sy)){
                lx[cnt] = cur_p.x;
                ly[cnt++] = cur_p.y;
                cur_p = fa[cur_p];
                // printf("x=%d y=%d cnt=%d\n",cur_p.x,cur_p.y,cnt);
            }
            // printf("cnt=%d\n\n",cnt);
            len[0] = cnt;
            ret[0] = 1;
            return ;

        }
        ret[0] = 0;
        return ;
        
    }
    

    void search_transpose(int *mat, int n, int m,
            int id,
            float tx, float ty,
            float sx, float sy,
            int cstate, int tstate, int bin,
            int *px, int *py, int num,
            float *lx, float *ly, int *lr, int *ld, int *len,
            int *ret
            )
    {
        // printf("before\n");
        memset(dist,-1,sizeof(dist));
        memset(sum,0,sizeof(sum));
        // printf("sx=%d sy=%d tx=%d ty=%d\n",sx,sy,tx,ty);
        priority_queue<node> pq;
        map<point, point> fa;

        float points[num*2], opoints[num*2];
        float radian = 2 * pi * cstate / bin;
        // printf("sx %d sy %d tx %d ty %d cstate %d tstate %d bin %d num %d\n",sx,sy,tx,ty,cstate,tstate,bin,num);
        float xmin,ymin,xmax,ymax;
        for(int i = 0;i < num;i++){
            // points[i<<1] = px[i];
            // points[i<<1|1] = py[i];
            opoints[i<<1] = px[i];
            opoints[i<<1|1] = py[i];
            // printf("px %f py %f %d %d\n",opoints[i<<1],opoints[i<<1|1],i<<1,i<<1|1);
        }
        rotate(opoints,points,num,radian,xmin,ymin,xmax,ymax);
        
        // printf("rote\n");
        for(int i = 0;i < num; i++){
            int x,y;
            // x = points[i<<1] + sx - 0.5*(xmax-xmin+1);
            // y = points[(i<<1)|1] + sy - 0.5*(ymax-ymin+1);
            x = points[i<<1] + sx;
            y = points[(i<<1)|1] + sy;
            // printf("x=%d y=%d %f %f\n",x,y,points[i<<1],points[(i<<1)|1]);
            if(x >= 0 && x < n && y >= 0 && y < m)
                mat[x*m+y] = 0;
        }
        

        
        sum[0][0] = mat[0];

        for(int i = 1; i < n; i++)
            sum[i][0] = sum[i-1][0] + mat[i*m];
        
        for(int i = 1; i < m; i++)
            sum[0][i] = sum[0][i-1] + mat[i];
        
        for(int i = 1; i < n; i++)
            for(int j = 1; j < m;j++)
                sum[i][j] = sum[i][j-1] + sum[i-1][j] - sum[i-1][j-1] + mat[i*m+j];        
        
        int f = 0;
        bool l = true;
        // printf("ok567\n");
        // printf("before\n");
        if(l){
            f = 0 + abs(sx-tx) + abs(sy-ty);
            pq.push(node(f,0,sx,sy,cstate));
            dist[int(sx)][int(sy)][cstate] = f;
            fa[point(sx,sy,cstate)] = point(sx,sy,cstate);
            float x,y;
            x = sx;
            y = sy;
            for(int ii = 1; ii < bin; ii++){
                int s = (cstate + ii) % bin;
                if(dist[int(x)][int(y)][s] != -1){
                    continue;
                }
                float ra = 2 * pi * s / bin;
                // printf("before check\n");
                bool push_flag = check(mat,opoints,num,ra,x,y,n,m);
                // printf("after check\n");
                // if(push_flag){
                //     printf("x=%d y=%d state=%d\n",x,y,s);
                // }
                if(push_flag){
                    dist[int(x)][int(y)][s] = f;
                    pq.push(node(f,0,x,y,s));
                    fa[point(x,y,s)] = point(x,y,cstate);
                }
                else{
                    break;
                }
            }

            for(int ii = 1;ii < bin;ii++){
                int s = (cstate - ii + bin) % bin;
                if(dist[int(x)][int(y)][s] != -1){
                    continue;
                }
                float ra = 2 * pi * s / bin;
                // printf("before check1\n");
                bool push_flag = check(mat,opoints,num,ra,x,y,n,m);
                // printf("after check1\n");
                // if(push_flag){
                //     printf("x=%d y=%d state=%d\n",x,y,s);
                // }
                if(push_flag){
                    dist[int(x)][int(y)][s] = f;
                    pq.push(node(f,0,x,y,s));
                    fa[point(x,y,s)] = point(x,y,cstate);
                }
                else{
                    break;
                }
            }
        }
        // printf("after\n");
        // printf("ok123\n");
        // printf("before\n");
        while(!pq.empty()){
            node a = pq.top();
            pq.pop();
            float x = a.x;
            float y = a.y;
            int d = a.d;
            int state = a.state;
            // printf("x=%d y=%d d=%d s=%d\n",a.x,a.y,a.d,a.state);
            if(fabs(x-tx) + fabs(y-ty) < EPS && state == tstate){
                break;
            }
            // printf("x=%d y=%d state=%d\n",x,y,state);

            float ra = 2 * pi * state / bin;
            
            for(int i = 0;i < 4; i++){
                float tmpx, tmpy;
                tmpx = x + _x[i];
                tmpy = y + _y[i];
                // printf("in1\n");
                if(tmpx < 0 || tmpx >= n || tmpy < 0 || tmpy >= m)
                    continue;

                bool succ = check(mat,opoints,num,ra,tmpx,tmpy,n,m);
                
                // printf("out1\n");
                // printf("yes\n");
                if(succ){
                    // printf("ohhh\n");
                    // printf("%d %d\n",tmpx,tmpy);
                    int f_ = d + 1 + fabs(tmpx - tx) + fabs(tmpy - ty);
                    if(dist[int(tmpx)][int(tmpy)][state] == -1 || dist[int(tmpx)][int(tmpy)][state] > f_){
                        dist[int(tmpx)][int(tmpy)][state] = f_;
                        pq.push(node(f_, d + 1, tmpx, tmpy, state));
                        fa[point(tmpx,tmpy,state)] = point(x, y, state);
                        // printf("in\n");
                        for(int ii = 1;ii < bin;ii++){
                            int s = (state + ii + bin) % bin;
                            if(dist[int(tmpx)][int(tmpy)][s] != -1){
                                continue;
                            }
                            float ra = 2 * pi * s / bin;
                            // printf("in2\n");
                            bool push_flag = check(mat,opoints,num,ra,tmpx,tmpy,n,m);
                            // printf("out2\n");
                            if(push_flag){
                                if(dist[int(tmpx)][int(tmpy)][s] == -1 || dist[int(tmpx)][int(tmpy)][s] > f_){
                                    dist[int(tmpx)][int(tmpy)][s] = f_;
                                    pq.push(node(f_,d+1,tmpx,tmpy,s));
                                    fa[point(tmpx,tmpy,s)] = point(x,y,state);
                                }
                            }
                            else{
                                break;
                            }
                        }
                        for(int ii = 1;ii < bin;ii++){
                            int s = (state - ii + bin) % bin;
                            if(dist[int(tmpx)][int(tmpy)][s] != -1){
                                continue;
                            }
                            float ra = 2 * pi * s / bin;
                            // printf("in3\n");
                            bool push_flag = check(mat,opoints,num,ra,tmpx,tmpy,n,m);
                            // printf("out3\n");
                            if(push_flag){
                                if(dist[int(tmpx)][int(tmpy)][s] == -1 || dist[int(tmpx)][int(tmpy)][s] > f_){
                                    dist[int(tmpx)][int(tmpy)][s] = f_;
                                    pq.push(node(f_,d+1,tmpx,tmpy,s));
                                    fa[point(tmpx,tmpy,s)] = point(x,y,state);
                                }
                            }
                            else{
                                break;
                            }
                        }
                        // printf("out\n");
                    }

                }
            }
        }
        // printf("after\n");

        // printf("ok\n");

        // for(int i = 0;i < num; i++){
        //     int x,y;
        //     x = points[i<<1] + sx;
        //     y = points[i<<1|1] + sy;
        //     mat[x*m+y] = id + 2;
        // }
        for(int i = 0;i < num; i++){
            int x,y;
            // x = points[i<<1] + sx - 0.5*(xmax-xmin+1);
            // y = points[(i<<1)|1] + sy - 0.5*(ymax-ymin+1);
            x = points[i<<1] + sx;
            y = points[(i<<1)|1] + sy;
            // printf("x=%d y=%d %f %f\n",x,y,points[i<<1],points[(i<<1)|1]);
            if(x >= 0 && x < n && y >= 0 && y < m)
                mat[x*m+y] = id + 2;
        }
        

        // delete points;
        bool flag = true;

        if(dist[int(tx)][int(ty)][tstate] == -1)
            flag = false;
        // printf("dis %d\n",dis[tx][ty]);
        if(flag){
            // printf("sx=%d sy=%d, tx=%d ty=%d\n",sx,sy,tx,ty);
            point cur_p = point(tx,ty,tstate);
            point start = point(sx,sy,cstate);
            // point start = point(sx,sy);
            int cnt = 0;
            // printf("begin\n");
            // printf("cur_p (%f %f %d) start (%f %f %d)\n",sx,sy,cstate,tx,ty,tstate);
            while(!(start == cur_p)){
                float x,y;
                x = cur_p.x;
                y = cur_p.y;
                lx[cnt] = x;
                ly[cnt] = y;
                lr[cnt] = cur_p.state;
                // printf("go!! x %d y %d state %d\n",x,y,cur_p.state);
                point fat = fa[cur_p];
                int dir = 0;
                for(int i = 1;i < bin;i++){
                    int s = (fat.state + i) % bin;
                    if(dist[int(x)][int(y)][s]==-1)
                        break;
                    if(s == cur_p.state){
                        dir = 0;
                        break;
                    }
                }
                for(int i = 1;i < bin;i++){
                    int s = (fat.state - i + bin) % bin;
                    if(dist[int(x)][int(y)][s]==-1)
                        break;
                    if(s == cur_p.state){
                        dir = 1;
                        break;
                    }
                }
                ld[cnt++] = dir;

                cur_p = fa[cur_p];
                // printf("x=%f y=%f cnt=%d\n",cur_p.x,cur_p.y,cnt);
            }
            // printf("end\n");
            // printf("cnt=%d\n\n",cnt);
            len[0] = cnt;
            ret[0] = 1;
            // printf("after 1\n");
            return ;

        }
        ret[0] = 0;
        // printf("after 2\n");
        return ;
        
    }
           
    void search_transpose_poly(int *mat, float *shapex, float *shapey, int *pn, int n, int m,
            int total, int id,
            float *tx_ar, float *ty_ar,
            float *sx_ar, float *sy_ar,
            int *cstate_ar, int *tstate_ar, int bin,
            // int *px, int *py, int num,
            float *lx, float *ly, int *lr, int *ld, int *len,
            int *ret
            )
    {
        // printf("before\n");
        memset(dist,-1,sizeof(dist));
        memset(sum,0,sizeof(sum));
        priority_queue<node> pq;
        map<point, point> fa;
        int amounts[total+1];
        amounts[0] = 0;

        for(int i = 1;i < total+1;i++)
            amounts[i] = amounts[i-1]+pn[i-1];
        

        sum[0][0] = mat[0];

        for(int i = 1; i < n; i++)
            sum[i][0] = sum[i-1][0] + mat[i*m];
        
        for(int i = 1; i < m; i++)
            sum[0][i] = sum[0][i-1] + mat[i];
        
        for(int i = 1; i < n; i++)
            for(int j = 1; j < m;j++)
                sum[i][j] = sum[i][j-1] + sum[i-1][j] - sum[i-1][j-1] + mat[i*m+j];        

        float tx,ty,sx,sy;
        int cstate, tstate;
        int num;

        num = pn[id];

        tx = tx_ar[id];
        ty = ty_ar[id];
        sx = sx_ar[id];
        sy = sy_ar[id];
        
        cstate = cstate_ar[id];
        tstate = tstate_ar[id];
        
        float points[bin][num*2], opoints[num*2];
        float shapes[amounts[total]*2], oshapes[amounts[total]*2];
        float radian = 2 * pi * cstate / bin;

        for(int i = 0;i < amounts[total];i++){
            oshapes[i<<1] = shapex[i];
            oshapes[i<<1|1] = shapey[i];
        }

        
        float xmin[bin],ymin[bin],xmax[bin],ymax[bin];

        for(int i = 0;i < total;i++){
            float ra = 2 * pi * cstate_ar[i] / bin;
            rotate(oshapes+amounts[i]*2,shapes+amounts[i]*2,pn[i],ra,xmin[0],ymin[0],xmax[0],ymax[0]);
            for(int j = 0; j < pn[i]; j++){
                shapes[(amounts[i]+j)<<1] += sx_ar[i];
                shapes[(amounts[i]+j)<<1|1] += sy_ar[i];
            }
        }

        for(int i = 0;i < num;i++){

            opoints[i<<1] = oshapes[(amounts[id]+i)<<1];
            opoints[i<<1|1] = oshapes[(amounts[id]+i)<<1|1];
            // printf("px %f py %f %d %d\n",opoints[i<<1],opoints[i<<1|1],i<<1,i<<1|1);
        }
        for(int i = 0;i < bin;i++)
            rotate(opoints,points[i],num, 2 * pi * i / bin, xmin[i],ymin[i],xmax[i],ymax[i]);
        
        int f = 0;
        bool l = true;
        // printf("ok567\n");
        // printf("before\n");
        if(l){
            f = 0 + abs(sx-tx) + abs(sy-ty);
            pq.push(node(f,0,sx,sy,cstate));
            dist[int(sx)][int(sy)][cstate] = f;
            fa[point(sx,sy,cstate)] = point(sx,sy,cstate);
            float x,y;
            x = sx;
            y = sy;
            for(int ii = 1; ii < bin; ii++){
                int s = (cstate + ii) % bin;
                if(dist[int(x)][int(y)][s] != -1){
                    continue;
                }
                float ra = 2 * pi * s / bin;
                // printf("before check\n");
                bool push_flag = check_poly_fast(mat, points[s],num,x,y,n,m,xmin[s],xmax[s],ymin[s],ymax[s]);
                // printf("after check\n");
                // if(push_flag){
                //     printf("x=%d y=%d state=%d\n",x,y,s);
                // }
                if(push_flag){
                    dist[int(x)][int(y)][s] = f;
                    pq.push(node(f,0,x,y,s));
                    fa[point(x,y,s)] = point(x,y,cstate);
                }
                else{
                    break;
                }
            }

            for(int ii = 1;ii < bin;ii++){
                int s = (cstate - ii + bin) % bin;
                if(dist[int(x)][int(y)][s] != -1){
                    continue;
                }
                float ra = 2 * pi * s / bin;
                // printf("before check1\n");
                // bool push_flag = check_poly(mat,shapes,opoints,amounts,pn,total,id,ra,x,y,n,m);
                bool push_flag = check_poly_fast(mat, points[s],num,x,y,n,m,xmin[s],xmax[s],ymin[s],ymax[s]);
                // printf("after check1\n");
                // if(push_flag){
                //     printf("x=%d y=%d state=%d\n",x,y,s);
                // }
                if(push_flag){
                    dist[int(x)][int(y)][s] = f;
                    pq.push(node(f,0,x,y,s));
                    fa[point(x,y,s)] = point(x,y,cstate);
                }
                else{
                    break;
                }
            }
        }
        // printf("after\n");
        // printf("ok123\n");
        // printf("before\n");
        // int count = 0;
        // printf("target tx=%f ty=%f tstate=%d\n",tx,ty,tstate);
        while(!pq.empty()){
            // count++;
            // if(count > 256*256*12){
            //     printf("what the fuck\n");
            // }
            node a = pq.top();
            pq.pop();
            float x = a.x;
            float y = a.y;
            int d = a.d;
            int state = a.state;
            // printf("x=%f y=%f d=%d s=%d\n",a.x,a.y,a.d,a.state);
            if(fabs(x-tx) + fabs(y-ty) < EPS && state == tstate){
                break;
            }
            // printf("x=%d y=%d state=%d\n",x,y,state);

            float ra = 2 * pi * state / bin;
            
            for(int i = 0;i < 4; i++){
                float tmpx, tmpy;
                tmpx = x + _x[i];
                tmpy = y + _y[i];
                // printf("in1\n");
                if(tmpx < 0 || tmpx >= n || tmpy < 0 || tmpy >= m)
                    continue;
                
                // bool succ = check_poly(mat,shapes,opoints,amounts,pn,total,id,ra,tmpx,tmpy,n,m);
                bool succ = check_poly_fast(mat,points[state],num,tmpx,tmpy,n,m,xmin[state],xmax[state],ymin[state],ymax[state]);
                // printf("tmpx %f tmpy %f succ %d\n",tmpx,tmpy,succ);
                // printf("out1\n");
                // printf("yes\n");
                if(succ){
                    // printf("ohhh\n");
                    // printf("%d %d\n",tmpx,tmpy);
                    int f_ = d + 1 + fabs(tmpx - tx) + fabs(tmpy - ty);
                    if(dist[int(tmpx)][int(tmpy)][state] == -1 || dist[int(tmpx)][int(tmpy)][state] > f_){
                        dist[int(tmpx)][int(tmpy)][state] = f_;
                        pq.push(node(f_, d + 1, tmpx, tmpy, state));
                        fa[point(tmpx,tmpy,state)] = point(x, y, state);
                        // printf("in\n");
                        for(int ii = 1;ii < bin;ii++){
                            int s = (state + ii + bin) % bin;
                            if(dist[int(tmpx)][int(tmpy)][s] != -1){
                                continue;
                            }
                            float ra = 2 * pi * s / bin;
                            // printf("in2\n");
                            // bool push_flag = check_poly(mat,shapes,opoints,amounts,pn,total,id,ra,tmpx,tmpy,n,m);
                            bool push_flag = check_poly_fast(mat, points[s],num,tmpx,tmpy,n,m,xmin[s],xmax[s],ymin[s],ymax[s]);
                            // printf("out2\n");
                            if(push_flag){
                                if(dist[int(tmpx)][int(tmpy)][s] == -1 || dist[int(tmpx)][int(tmpy)][s] > f_){
                                    dist[int(tmpx)][int(tmpy)][s] = f_;
                                    pq.push(node(f_,d+1,tmpx,tmpy,s));
                                    fa[point(tmpx,tmpy,s)] = point(x,y,state);
                                }
                            }
                            else{
                                break;
                            }
                        }
                        for(int ii = 1;ii < bin;ii++){
                            int s = (state - ii + bin) % bin;
                            if(dist[int(tmpx)][int(tmpy)][s] != -1){
                                continue;
                            }
                            float ra = 2 * pi * s / bin;
                            // printf("in3\n");
                            // bool push_flag = check_poly(mat,shapes,opoints,amounts,pn,total,id,ra,tmpx,tmpy,n,m);
                            bool push_flag = check_poly_fast(mat, points[s],num,tmpx,tmpy,n,m,xmin[s],xmax[s],ymin[s],ymax[s]);
                            // printf("out3\n");
                            if(push_flag){
                                if(dist[int(tmpx)][int(tmpy)][s] == -1 || dist[int(tmpx)][int(tmpy)][s] > f_){
                                    dist[int(tmpx)][int(tmpy)][s] = f_;
                                    pq.push(node(f_,d+1,tmpx,tmpy,s));
                                    fa[point(tmpx,tmpy,s)] = point(x,y,state);
                                }
                            }
                            else{
                                break;
                            }
                        }
                        // printf("out\n");
                    }

                }
            }
        }
        
        // delete points;
        bool flag = true;

        if(dist[int(tx)][int(ty)][tstate] == -1)
            flag = false;
        // printf("dis %d\n",dis[tx][ty]);
        if(flag){
            // printf("sx=%d sy=%d, tx=%d ty=%d\n",sx,sy,tx,ty);
            point cur_p = point(tx,ty,tstate);
            point start = point(sx,sy,cstate);
            // point start = point(sx,sy);
            int cnt = 0;
            // printf("begin\n");
            // printf("cur_p (%f %f %d) start (%f %f %d)\n",sx,sy,cstate,tx,ty,tstate);
            while(!(start == cur_p)){
                float x,y;
                x = cur_p.x;
                y = cur_p.y;
                lx[cnt] = x;
                ly[cnt] = y;
                lr[cnt] = cur_p.state;
                // printf("go!! x %d y %d state %d\n",x,y,cur_p.state);
                point fat = fa[cur_p];
                int dir = 0;
                for(int i = 1;i < bin;i++){
                    int s = (fat.state + i) % bin;
                    if(dist[int(x)][int(y)][s]==-1)
                        break;
                    if(s == cur_p.state){
                        dir = 0;
                        break;
                    }
                }
                for(int i = 1;i < bin;i++){
                    int s = (fat.state - i + bin) % bin;
                    if(dist[int(x)][int(y)][s]==-1)
                        break;
                    if(s == cur_p.state){
                        dir = 1;
                        break;
                    }
                }
                ld[cnt++] = dir;

                cur_p = fa[cur_p];
                // printf("x=%f y=%f cnt=%d\n",cur_p.x,cur_p.y,cnt);
            }
            // printf("end\n");
            // printf("cnt=%d\n\n",cnt);
            len[0] = cnt;
            ret[0] = 1;
            // printf("after 1\n");
            return ;

        }
        ret[0] = 0;
        // printf("after 2\n");
        return ;
        
    }


    void translate(int *mat, float *shapex, float *shapey, int *pn, int n, int m,
            int total, int id, int direction,
            float *sx_ar, float *sy_ar,
            int *cstate_ar, int *tstate_ar, int bin,
            int *step
    )
    {
        int amounts[total+1];
        amounts[0] = 0;
        int _x[4] = {-1,1,0,0};
        int _y[4] = {0,0,-1,1};
        int num = pn[id];

        for(int i = 1; i < total+1; i++)
            amounts[i] = amounts[i-1] + pn[i-1];

        float shapes[num*2], oshapes[num*2];
        float xmin,ymin,xmax,ymax;
        vector<point> vpa;

        for(int i = 0;i < num;i++){
            oshapes[i<<1] = shapex[i+amounts[id]];
            oshapes[i<<1|1] = shapey[i+amounts[id]];
        }

        
        float ra = 2 * pi * cstate_ar[id] / bin;
        rotate(oshapes,shapes,num,ra,xmin,ymin,xmax,ymax);
            
            
        for(int j = 0;j < num; j++)
            vpa.push_back(point(shapes[j<<1],shapes[j<<1|1]));
            
        float xx,yy;
        xx = sx_ar[id];
        yy = sy_ar[id];

        // for(int i = 0;i < vpa.size();i++){
        //     printf("x=%f y=%f\n",xx+vpa[i].x, yy+vpa[i].y);
        // }
        
        int steps = 1;
        
        while(true){
            // printf("==========\n");
            float x,y;
            x = xx + steps * _x[direction];
            y = yy + steps * _y[direction];
            if(x + xmin < 0 || x + xmax >= n || y+ymin < 0 || y+ymax >= m ){
                steps -= 1;
                // printf("what\n");
                break;
            }

            bool flag = true;
            vector<point> tpa = vpa;
            for(int i = 0;i < tpa.size();i++){
                tpa[i] = tpa[i] + point(x,y);
                int xx_,yy_;
                xx_ = tpa[i].x + EPS;
                yy_ = tpa[i].y + EPS;
                // printf("%d %d x=%f y=%f\n",xx_,yy_,tpa[i].x + EPS,tpa[i].y + EPS);
                // if(xx_ < 0 || xx_ >= n || yy_ < 0 || yy_ >= m){
                //     printf("s=%d\n",steps);
                //     flag = false;
                //     break;
                // }
                if(mat[xx_*m+yy_] != 0 && mat[xx_*m+yy_] != id + 2){
                    // if(id == 8 && direction == 1){
                    //     printf("xx %d yy %d, id %d\n",xx,yy,mat[xx*m+yy]);
                    // }

                    flag = false;
                    break;
                }
                // printf("%f %f\n",tpa[i].x,tpa[i].y);
            }

            for(int i = 0; i < tpa.size() && flag; i++){
                point pa,pb;
                pa = tpa[i];
                pb = tpa[(i+1)%tpa.size()];
                int loop = max(fabs(pb.x - pa.x) ,fabs(pb.y-pa.y)) + EPS + 1;
                for(int j = 0;j < loop + 1; j++){
                    point pt = (1.0-1.0*j/loop)*pa+(1.0*j/loop)*pb;
                    int x,y;
                    x = pt.x + EPS;
                    y = pt.y + EPS;
                    // if(id == 8 && direction == 1)
                    //     printf("x=%f y=%f\n",pt.x,pt.y);
                    if(mat[x*m+y] != 0 && mat[x*m+y] != id + 2){
                        // if(id == 8 && direction == 1){
                        //     printf("22 xx %d yy %d, id %d\n",x,y,mat[x*m+y]);
                        // }
                        flag = false;
                        break;
                    }
                }
            }


            if(!flag){
                steps -= 1;
                break;
            }
            // printf("=============\n");
            // for(int i = 0;i < shapes_v[1].size();i++){
            //     printf("%f %f\n",shapes_v[1][i].x,shapes_v[1][i].y);
            // }
            

            // for(int i = 0;i < total;i++){
            //     if(i != id){
            //         // float val = NPIA(tpa,shapes_v[i]);
                    
            //         // if(val > EPS){
            //         if(convex_cross(tpa,shapes_v[i])){
            //             // printf("id %d val %f\n",i,val);
            //             flag = false;
            //             break;
            //         }
            //     }
            // }
            // if(!flag){
            //     steps -= 1;
            //     break;
            // }

            steps += 1;
        }
        step[0] = steps;
        // printf("steps:%d\n",steps);
    }

    struct thread_data{
        int *mat;
        float *points; 
        int num; 
        float x; 
        float y; 
        int n; 
        int m;
        float xmin;
        float xmax;
        float ymin;
        float ymax;
        bool *flag;
        int f;
        int s;
        thread_data(){}
        thread_data(int *mat, float *points, int num, float x, float y, int n, int m, float xmin, float xmax, float ymin, float ymax, bool *flag, int f, int s):mat(mat),points(points),num(num),x(x),y(y),n(n),m(m),xmin(xmin),xmax(xmax),ymin(ymin),ymax(ymax),flag(flag),f(f),s(s){}
    };

    void *pthread_check(void *args){
        struct thread_data * data;
        data = (struct thread_data *)args;
        int *mat;
        float *points; 
        int num; 
        float x; 
        float y; 
        int n; 
        int m;
        float xmin;
        float xmax;
        float ymin;
        float ymax;
        bool *flag;
        
        mat = data->mat;
        points = data->points;
        num = data->num;
        x = data->x;
        y = data->y;
        n = data->n;
        m = data->m;
        xmin = data->xmin;
        xmax = data->xmax;
        ymin = data->ymin;
        ymax = data->ymax;
        flag = data->flag;
        *flag = check_poly_fast(mat, points, num, x, y, n, m, xmin, xmax, ymin, ymax);
    }

    void search_transpose_poly_rot(int *mat, float *shapex, float *shapey, int *pn, int n, int m,
            int total, int id,
            float *tx_ar, float *ty_ar,
            float *sx_ar, float *sy_ar,
            int *cstate_ar, int *tstate_ar, int bin,
            // int *px, int *py, int num,
            float *lx, float *ly, int *lr, int *ld, int *len,
            int *ret
            )
    {
        // printf("before\n");
        memset(dist,-1,sizeof(dist));
        memset(sum,0,sizeof(sum));
        priority_queue<node> pq;
        map<point, point> fa;
        int amounts[total+1];
        amounts[0] = 0;

        for(int i = 1;i < total+1;i++)
            amounts[i] = amounts[i-1]+pn[i-1];
        

        sum[0][0] = mat[0];

        for(int i = 1; i < n; i++)
            sum[i][0] = sum[i-1][0] + mat[i*m];
        
        for(int i = 1; i < m; i++)
            sum[0][i] = sum[0][i-1] + mat[i];
        
        for(int i = 1; i < n; i++)
            for(int j = 1; j < m;j++)
                sum[i][j] = sum[i][j-1] + sum[i-1][j] - sum[i-1][j-1] + mat[i*m+j];        

        float tx,ty,sx,sy;
        int cstate, tstate;
        int num;

        num = pn[id];

        tx = tx_ar[id];
        ty = ty_ar[id];
        sx = sx_ar[id];
        sy = sy_ar[id];
        
        cstate = cstate_ar[id];
        tstate = tstate_ar[id];
        
        float points[bin][num*2], opoints[num*2];
        float shapes[amounts[total]*2], oshapes[amounts[total]*2];
        float radian = 2 * pi * cstate / bin;

        for(int i = 0;i < amounts[total];i++){
            oshapes[i<<1] = shapex[i];
            oshapes[i<<1|1] = shapey[i];
        }

        
        float xmin[bin],ymin[bin],xmax[bin],ymax[bin];

        for(int i = 0;i < total;i++){
            float ra = 2 * pi * cstate_ar[i] / bin;
            rotate(oshapes+amounts[i]*2,shapes+amounts[i]*2,pn[i],ra,xmin[0],ymin[0],xmax[0],ymax[0]);
            for(int j = 0; j < pn[i]; j++){
                shapes[(amounts[i]+j)<<1] += sx_ar[i];
                shapes[(amounts[i]+j)<<1|1] += sy_ar[i];
            }
        }

        for(int i = 0;i < num;i++){

            opoints[i<<1] = oshapes[(amounts[id]+i)<<1];
            opoints[i<<1|1] = oshapes[(amounts[id]+i)<<1|1];
            // printf("px %f py %f %d %d\n",opoints[i<<1],opoints[i<<1|1],i<<1,i<<1|1);
        }
        for(int i = 0;i < bin;i++)
            rotate(opoints,points[i],num, 2 * pi * i / bin, xmin[i],ymin[i],xmax[i],ymax[i]);
        
        int f = 0;
        bool l = true;
        // printf("ok567\n");
        // printf("before\n");
        if(l){
            f = 0 + abs(sx-tx) + abs(sy-ty);
            pq.push(node(f,0,sx,sy,cstate));
            dist[int(sx)][int(sy)][cstate] = f;
            fa[point(sx,sy,cstate)] = point(sx,sy,cstate);
            
        }
        // printf("after\n");
        // printf("ok123\n");
        // printf("before\n");
        // int count = 0;
        // printf("target tx=%f ty=%f tstate=%d\n",tx,ty,tstate);
        while(!pq.empty()){
            // count++;
            // if(count > 256*256*12){
            //     printf("what the fuck\n");
            // }
            node a = pq.top();
            pq.pop();
            float x = a.x;
            float y = a.y;
            int d = a.d;
            int state = a.state;
            // printf("x=%f y=%f d=%d s=%d\n",a.x,a.y,a.d,a.state);
            if(fabs(x-tx) + fabs(y-ty) < EPS && state == tstate){
                break;
            }
            // printf("x=%d y=%d state=%d\n",x,y,state);

            float ra = 2 * pi * state / bin;
            bool flags[6];
            struct thread_data params[6];
            pthread_t threads[6];
            memset(flags,0,sizeof(flags));
            int cnt_id = 0;
            
            for(int i = 0;i < 4; i++){
                float tmpx, tmpy;
                tmpx = x + _x[i];
                tmpy = y + _y[i];
                // printf("in1\n");
                if(tmpx < 0 || tmpx >= n || tmpy < 0 || tmpy >= m)
                    continue;
                
                int f_ = d + 1 + fabs(tmpx - tx) + fabs(tmpy - ty);
                params[cnt_id] = thread_data(mat,points[state],num,tmpx,tmpy,n,m,xmin[state],xmax[state],ymin[state],ymax[state],&flags[cnt_id],f_,state);
                pthread_create(&threads[cnt_id], NULL, pthread_check, (void *)&params[cnt_id]);
                cnt_id++;
                // // bool succ = check_poly(mat,shapes,opoints,amounts,pn,total,id,ra,tmpx,tmpy,n,m);
                // bool succ = check_poly_fast(mat,points[state],num,tmpx,tmpy,n,m,xmin[state],xmax[state],ymin[state],ymax[state]);
                // // printf("tmpx %f tmpy %f succ %d\n",tmpx,tmpy,succ);
                // // printf("out1\n");
                // // printf("yes\n");
                // if(succ){
                //     // printf("ohhh\n");
                //     // printf("%d %d\n",tmpx,tmpy);
                //     int f_ = d + 1 + fabs(tmpx - tx) + fabs(tmpy - ty);
                //     if(dist[int(tmpx)][int(tmpy)][state] == -1 || dist[int(tmpx)][int(tmpy)][state] > f_){
                //         dist[int(tmpx)][int(tmpy)][state] = f_;
                //         pq.push(node(f_, d + 1, tmpx, tmpy, state));
                //         fa[point(tmpx,tmpy,state)] = point(x, y, state);
                //         // printf("in\n");
                //         // printf("out\n");
                //     }

                // }
            }
            
            int s = (state + 1 + bin) % bin;
            int f_ = a.f;
            ra = 2 * pi * s / bin;

            params[cnt_id] = thread_data(mat,points[s],num,x,y,n,m,xmin[s],xmax[s],ymin[s],ymax[s],&flags[cnt_id],f_,s);
            pthread_create(&threads[cnt_id], NULL, pthread_check, (void*)&params[cnt_id]);
            cnt_id++;
            // bool push_flag = check_poly_fast(mat, points[s],num,x,y,n,m,xmin[s],xmax[s],ymin[s],ymax[s]);
            // // printf("out2\n");
            // if(push_flag){
            //     if(dist[int(x)][int(y)][s] == -1 || dist[int(x)][int(y)][s] > f_){
            //         dist[int(x)][int(y)][s] = f_;
            //         pq.push(node(f_,d+1,x,y,s));
            //         fa[point(x,y,s)] = point(x,y,state);
            //     }
            // }
            
           
            
            
           
            s = (state - 1 + bin) % bin;
            
            ra = 2 * pi * s / bin;

            params[cnt_id] = thread_data(mat,points[s],num,x,y,n,m,xmin[s],xmax[s],ymin[s],ymax[s],&flags[cnt_id],f_,s);
            pthread_create(&threads[cnt_id], NULL, pthread_check, (void*)&params[cnt_id]);
            cnt_id++;

            for(int i = 0;i < 6; i++){
                if(flags[i]){
                    x = params[i].x;
                    y = params[i].y;
                    f_ = params[i].f;
                    s = params[i].s;
                    if(dist[int(x)][int(y)][s] == -1 || dist[int(x)][int(y)][s] > f_){
                        dist[int(x)][int(y)][s] = f_;
                        pq.push(node(f_,d+1,x,y,s));
                        fa[point(x,y,s)] = point(x,y,state);
                    }
                    
                }
            }
            // push_flag = check_poly_fast(mat, points[s],num,x,y,n,m,xmin[s],xmax[s],ymin[s],ymax[s]);
            // // printf("out3\n");
            // if(push_flag){
            //     if(dist[int(x)][int(y)][s] == -1 || dist[int(x)][int(y)][s] > f_){
            //         dist[int(x)][int(y)][s] = f_;
            //         pq.push(node(f_,d+1,x,y,s));
            //         fa[point(x,y,s)] = point(x,y,state);
            //     }
            // }
            
        }
        
        // delete points;
        bool flag = true;

        if(dist[int(tx)][int(ty)][tstate] == -1)
            flag = false;
        // printf("dis %d\n",dis[tx][ty]);
        if(flag){
            // printf("sx=%d sy=%d, tx=%d ty=%d\n",sx,sy,tx,ty);
            point cur_p = point(tx,ty,tstate);
            point start = point(sx,sy,cstate);
            // point start = point(sx,sy);
            int cnt = 0;
            // printf("begin\n");
            // printf("cur_p (%f %f %d) start (%f %f %d)\n",sx,sy,cstate,tx,ty,tstate);
            while(!(start == cur_p)){
                float x,y;
                x = cur_p.x;
                y = cur_p.y;
                lx[cnt] = x;
                ly[cnt] = y;
                lr[cnt] = cur_p.state;
                // printf("go!! x %d y %d state %d\n",x,y,cur_p.state);
                point fat = fa[cur_p];
                int dir = 0;
                for(int i = 1;i < bin;i++){
                    int s = (fat.state + i) % bin;
                    if(dist[int(x)][int(y)][s]==-1)
                        break;
                    if(s == cur_p.state){
                        dir = 0;
                        break;
                    }
                }
                for(int i = 1;i < bin;i++){
                    int s = (fat.state - i + bin) % bin;
                    if(dist[int(x)][int(y)][s]==-1)
                        break;
                    if(s == cur_p.state){
                        dir = 1;
                        break;
                    }
                }
                ld[cnt++] = dir;

                cur_p = fa[cur_p];
                // printf("x=%f y=%f cnt=%d\n",cur_p.x,cur_p.y,cnt);
            }
            // printf("end\n");
            // printf("cnt=%d\n\n",cnt);
            len[0] = cnt;
            ret[0] = 1;
            // printf("after 1\n");
            return ;

        }
        ret[0] = 0;
        // printf("after 2\n");
        return ;
        
    }

    void search_transpose_poly_terrain(int *mat, int *terrain, float *shapex, float *shapey, int *pn, int n, int m,
        int total, int id,
        float *tx_ar, float *ty_ar,
        float *sx_ar, float *sy_ar,
        int *cstate_ar, int *tstate_ar, int bin,
        // int *px, int *py, int num,
        float *lx, float *ly, int *lr, int *ld, int *len,
        int *ret
        )
    {
        // printf("before\n");
        memset(dist,-1,sizeof(dist));
        memset(sum,0,sizeof(sum));
        priority_queue<node> pq;
        map<point, point> fa;
        int amounts[total+1];
        amounts[0] = 0;

        for(int i = 1;i < total+1;i++)
            amounts[i] = amounts[i-1]+pn[i-1];
        

        sum[0][0] = mat[0];

        for(int i = 1; i < n; i++)
            sum[i][0] = sum[i-1][0] + mat[i*m];
        
        for(int i = 1; i < m; i++)
            sum[0][i] = sum[0][i-1] + mat[i];
        
        for(int i = 1; i < n; i++)
            for(int j = 1; j < m;j++)
                sum[i][j] = sum[i][j-1] + sum[i-1][j] - sum[i-1][j-1] + mat[i*m+j];        

        float tx,ty,sx,sy;
        int cstate, tstate;
        int num;

        num = pn[id];

        tx = tx_ar[id];
        ty = ty_ar[id];
        sx = sx_ar[id];
        sy = sy_ar[id];
        
        cstate = cstate_ar[id];
        tstate = tstate_ar[id];
        
        float points[bin][num*2], opoints[num*2];
        float shapes[amounts[total]*2], oshapes[amounts[total]*2];
        float radian = 2 * pi * cstate / bin;

        for(int i = 0;i < amounts[total];i++){
            oshapes[i<<1] = shapex[i];
            oshapes[i<<1|1] = shapey[i];
        }

        
        float xmin[bin],ymin[bin],xmax[bin],ymax[bin];

        for(int i = 0;i < total;i++){
            float ra = 2 * pi * cstate_ar[i] / bin;
            rotate(oshapes+amounts[i]*2,shapes+amounts[i]*2,pn[i],ra,xmin[0],ymin[0],xmax[0],ymax[0]);
            for(int j = 0; j < pn[i]; j++){
                shapes[(amounts[i]+j)<<1] += sx_ar[i];
                shapes[(amounts[i]+j)<<1|1] += sy_ar[i];
            }
        }

        for(int i = 0;i < num;i++){

            opoints[i<<1] = oshapes[(amounts[id]+i)<<1];
            opoints[i<<1|1] = oshapes[(amounts[id]+i)<<1|1];
            // printf("px %f py %f %d %d\n",opoints[i<<1],opoints[i<<1|1],i<<1,i<<1|1);
        }
        for(int i = 0;i < bin;i++)
            rotate(opoints,points[i],num, 2 * pi * i / bin, xmin[i],ymin[i],xmax[i],ymax[i]);
        
        int f = 0;
        bool l = true;
        // printf("ok567\n");
        // printf("before\n");
        if(l){
            f = 0 + abs(sx-tx) + abs(sy-ty);
            pq.push(node(f,0,sx,sy,cstate));
            dist[int(sx)][int(sy)][cstate] = f;
            fa[point(sx,sy,cstate)] = point(sx,sy,cstate);
            float x,y;
            x = sx;
            y = sy;
            for(int ii = 1; ii < bin; ii++){
                int s = (cstate + ii) % bin;
                int pres = (cstate + ii - 1) % bin;
                if(dist[int(x)][int(y)][s] != -1){
                    continue;
                }
                float ra = 2 * pi * s / bin;
                // printf("before check\n");
                bool push_flag = check_poly_fast_terrain(mat, terrain, points[s], points[pres],num,x,y,n,m,xmin[s],xmax[s],ymin[s],ymax[s],-1);
                // printf("after check\n");
                // if(push_flag){
                //     printf("x=%d y=%d state=%d\n",x,y,s);
                // }
                if(push_flag){
                    dist[int(x)][int(y)][s] = f;
                    pq.push(node(f,0,x,y,s));
                    fa[point(x,y,s)] = point(x,y,cstate);
                }
                else{
                    break;
                }
            }

            for(int ii = 1;ii < bin;ii++){
                int s = (cstate - ii + bin) % bin;
                int pres = (cstate - ii + bin + 1) % bin;
                if(dist[int(x)][int(y)][s] != -1){
                    continue;
                }
                float ra = 2 * pi * s / bin;
                // printf("before check1\n");
                // bool push_flag = check_poly(mat,shapes,opoints,amounts,pn,total,id,ra,x,y,n,m);
                bool push_flag = check_poly_fast_terrain(mat, terrain, points[s], points[pres],num,x,y,n,m,xmin[s],xmax[s],ymin[s],ymax[s],-1);
                // printf("after check1\n");
                // if(push_flag){
                //     printf("x=%d y=%d state=%d\n",x,y,s);
                // }
                if(push_flag){
                    dist[int(x)][int(y)][s] = f;
                    pq.push(node(f,0,x,y,s));
                    fa[point(x,y,s)] = point(x,y,cstate);
                }
                else{
                    break;
                }
            }
        }
        // printf("after\n");
        // printf("ok123\n");
        // printf("before\n");
        // int count = 0;
        // printf("target tx=%f ty=%f tstate=%d\n",tx,ty,tstate);
        while(!pq.empty()){
            // count++;
            // if(count > 256*256*12){
            //     printf("what the fuck\n");
            // }
            node a = pq.top();
            pq.pop();
            float x = a.x;
            float y = a.y;
            int d = a.d;
            int state = a.state;
            // printf("x=%f y=%f d=%d s=%d\n",a.x,a.y,a.d,a.state);
            if(fabs(x-tx) + fabs(y-ty) < EPS && state == tstate){
                break;
            }
            // printf("x=%d y=%d state=%d\n",x,y,state);

            float ra = 2 * pi * state / bin;
            
            for(int i = 0;i < 4; i++){
                float tmpx, tmpy;
                tmpx = x + _x[i];
                tmpy = y + _y[i];
                // printf("in1\n");
                if(tmpx < 0 || tmpx >= n || tmpy < 0 || tmpy >= m)
                    continue;
                
                // bool succ = check_poly(mat,shapes,opoints,amounts,pn,total,id,ra,tmpx,tmpy,n,m);
                bool succ = check_poly_fast_terrain(mat, terrain, points[state],points[state],num,tmpx,tmpy,n,m,xmin[state],xmax[state],ymin[state],ymax[state],i);
                // printf("tmpx %f tmpy %f succ %d\n",tmpx,tmpy,succ);
                // printf("out1\n");
                // printf("yes\n");
                if(succ){
                    // printf("ohhh\n");
                    // printf("%d %d\n",tmpx,tmpy);
                    int f_ = d + 1 + fabs(tmpx - tx) + fabs(tmpy - ty);
                    if(dist[int(tmpx)][int(tmpy)][state] == -1 || dist[int(tmpx)][int(tmpy)][state] > f_){
                        dist[int(tmpx)][int(tmpy)][state] = f_;
                        pq.push(node(f_, d + 1, tmpx, tmpy, state));
                        fa[point(tmpx,tmpy,state)] = point(x, y, state);
                        // printf("in\n");
                        for(int ii = 1;ii < bin;ii++){
                            int s = (state + ii + bin) % bin;
                            int pres = (state + ii + bin - i) % bin;
                            if(dist[int(tmpx)][int(tmpy)][s] != -1){
                                continue;
                            }
                            float ra = 2 * pi * s / bin;
                            // printf("in2\n");
                            // bool push_flag = check_poly(mat,shapes,opoints,amounts,pn,total,id,ra,tmpx,tmpy,n,m);
                            bool push_flag = check_poly_fast_terrain(mat, terrain, points[s], points[pres],num,tmpx,tmpy,n,m,xmin[s],xmax[s],ymin[s],ymax[s],-1);
                            // printf("out2\n");
                            if(push_flag){
                                if(dist[int(tmpx)][int(tmpy)][s] == -1 || dist[int(tmpx)][int(tmpy)][s] > f_){
                                    dist[int(tmpx)][int(tmpy)][s] = f_;
                                    pq.push(node(f_,d+1,tmpx,tmpy,s));
                                    fa[point(tmpx,tmpy,s)] = point(x,y,state);
                                }
                            }
                            else{
                                break;
                            }
                        }
                        for(int ii = 1;ii < bin;ii++){
                            int s = (state - ii + bin) % bin;
                            int pres = (state - ii + bin + 1) % bin;
                            if(dist[int(tmpx)][int(tmpy)][s] != -1){
                                continue;
                            }
                            float ra = 2 * pi * s / bin;
                            // printf("in3\n");
                            // bool push_flag = check_poly(mat,shapes,opoints,amounts,pn,total,id,ra,tmpx,tmpy,n,m);
                            bool push_flag = check_poly_fast_terrain(mat, terrain, points[s],points[pres],num,tmpx,tmpy,n,m,xmin[s],xmax[s],ymin[s],ymax[s],-1);
                            // printf("out3\n");
                            if(push_flag){
                                if(dist[int(tmpx)][int(tmpy)][s] == -1 || dist[int(tmpx)][int(tmpy)][s] > f_){
                                    dist[int(tmpx)][int(tmpy)][s] = f_;
                                    pq.push(node(f_,d+1,tmpx,tmpy,s));
                                    fa[point(tmpx,tmpy,s)] = point(x,y,state);
                                }
                            }
                            else{
                                break;
                            }
                        }
                        // printf("out\n");
                    }

                }
            }
        }
        
        // delete points;
        bool flag = true;

        if(dist[int(tx)][int(ty)][tstate] == -1)
            flag = false;
        // printf("dis %d\n",dis[tx][ty]);
        if(flag){
            // printf("sx=%d sy=%d, tx=%d ty=%d\n",sx,sy,tx,ty);
            point cur_p = point(tx,ty,tstate);
            point start = point(sx,sy,cstate);
            // point start = point(sx,sy);
            int cnt = 0;
            // printf("begin\n");
            // printf("cur_p (%f %f %d) start (%f %f %d)\n",sx,sy,cstate,tx,ty,tstate);
            while(!(start == cur_p)){
                float x,y;
                x = cur_p.x;
                y = cur_p.y;
                lx[cnt] = x;
                ly[cnt] = y;
                lr[cnt] = cur_p.state;
                // printf("go!! x %d y %d state %d\n",x,y,cur_p.state);
                point fat = fa[cur_p];
                int dir = 0;
                for(int i = 1;i < bin;i++){
                    int s = (fat.state + i) % bin;
                    if(dist[int(x)][int(y)][s]==-1)
                        break;
                    if(s == cur_p.state){
                        dir = 0;
                        break;
                    }
                }
                for(int i = 1;i < bin;i++){
                    int s = (fat.state - i + bin) % bin;
                    if(dist[int(x)][int(y)][s]==-1)
                        break;
                    if(s == cur_p.state){
                        dir = 1;
                        break;
                    }
                }
                ld[cnt++] = dir;

                cur_p = fa[cur_p];
                // printf("x=%f y=%f cnt=%d\n",cur_p.x,cur_p.y,cnt);
            }
            // printf("end\n");
            // printf("cnt=%d\n\n",cnt);
            len[0] = cnt;
            ret[0] = 1;
            // printf("after 1\n");
            return ;

        }
        ret[0] = 0;
        // printf("after 2\n");
        return ;
        
    }

    void translate_terrain(int *mat, int *terrain, float *shapex, float *shapey, int *pn, int n, int m,
            int total, int id, int direction,
            float *sx_ar, float *sy_ar,
            int *cstate_ar, int *tstate_ar, int bin,
            int *step
    )
    {
        int amounts[total+1];
        amounts[0] = 0;
        int _x[4] = {-1,1,0,0};
        int _y[4] = {0,0,-1,1};
        int num = pn[id];

        for(int i = 1; i < total+1; i++)
            amounts[i] = amounts[i-1] + pn[i-1];

        float shapes[num*2], oshapes[num*2];
        float xmin,ymin,xmax,ymax;
        vector<point> vpa;

        for(int i = 0;i < num;i++){
            oshapes[i<<1] = shapex[i+amounts[id]];
            oshapes[i<<1|1] = shapey[i+amounts[id]];
        }

        
        float ra = 2 * pi * cstate_ar[id] / bin;
        rotate(oshapes,shapes,num,ra,xmin,ymin,xmax,ymax);
            
            
        for(int j = 0;j < num; j++)
            vpa.push_back(point(shapes[j<<1],shapes[j<<1|1]));
            
        float xx,yy;
        xx = sx_ar[id];
        yy = sy_ar[id];
        
        int steps = 1;
        
        while(true){
            float x,y;
            x = xx + steps * _x[direction];
            y = yy + steps * _y[direction];
            if(x < 0 || x > n || y < 0 || y > m){
                steps -= 1;
                // printf("what\n");
                break;
            }

            bool flag = true;
            vector<point> tpa = vpa;
            for(int i = 0;i < tpa.size();i++){
                tpa[i] = tpa[i] + point(x,y);
                int xx,yy;
                xx = tpa[i].x + EPS;
                yy = tpa[i].y + EPS;

                if(mat[xx*m+yy] != 0 && mat[xx*m+yy] != id+2){
                    // if(id == 8 && direction == 1){
                    //     printf("xx %d yy %d, id %d\n",xx,yy,mat[xx*m+yy]);
                    // }

                    flag = false;
                    break;
                }
                // printf("%f %f\n",tpa[i].x,tpa[i].y);
            }

            for(int i = 0; i < tpa.size() && flag; i++){
                point pa,pb;
                pa = tpa[i];
                pb = tpa[(i+1)%tpa.size()];
                int loop = max(fabs(pb.x - pa.x) ,fabs(pb.y-pa.y)) + EPS + 1;
                for(int j = 0;j < loop + 1; j++){
                    point pt = (1.0-1.0*j/loop)*pa+(1.0*j/loop)*pb;
                    int x,y,dx,dy;
                    x = pt.x + EPS;
                    y = pt.y + EPS;
                    dx = x - _x[direction];
                    dy = y - _y[direction];
                    // if(id == 8 && direction == 1)
                    //     printf("x=%f y=%f\n",pt.x,pt.y);
                    if(mat[x*m+y] != 0 && mat[x*m+y] != id + 2 && abs(terrain[x*m+y]-terrain[dx*m+dy]) < 10){
                        // if(id == 8 && direction == 1){
                        //     printf("22 xx %d yy %d, id %d\n",x,y,mat[x*m+y]);
                        // }
                        flag = false;
                        break;
                    }
                }
            }


            if(!flag){
                steps -= 1;
                break;
            }
            // printf("=============\n");
            // for(int i = 0;i < shapes_v[1].size();i++){
            //     printf("%f %f\n",shapes_v[1][i].x,shapes_v[1][i].y);
            // }
            

            // for(int i = 0;i < total;i++){
            //     if(i != id){
            //         // float val = NPIA(tpa,shapes_v[i]);
                    
            //         // if(val > EPS){
            //         if(convex_cross(tpa,shapes_v[i])){
            //             // printf("id %d val %f\n",i,val);
            //             flag = false;
            //             break;
            //         }
            //     }
            // }
            // if(!flag){
            //     steps -= 1;
            //     break;
            // }

            steps += 1;
        }
        step[0] = steps;
        // printf("steps:%d\n",steps);
    }
    
    // void state(int ){

    // }
}


int main(){
    vector<point> ps,pb;
    // ps.push_back(point(-1,0));
    // ps.push_back(point(1,0));
    // ps.push_back(point(-0.5,-1));
    // ps.push_back(point(1,2));
    // // ps.push_back(point(2,2));
    // ps.push_back(point(2,1));
    // ps.push_back(point(2,-1));
    // // ps.push_back(point(2,-2));
    // ps.push_back(point(1,-2));
    // ps.push_back(point(-1,-2));
    // // ps.push_back(point(-2,-2));
    // ps.push_back(point(-2,-1));
    // ps.push_back(point(-2,1));
    // // ps.push_back(point(-2,2));
    // ps.push_back(point(-1,2));

    ps.push_back(point(0,2));
    ps.push_back(point(2,2));
    ps.push_back(point(0,0));
    ps.push_back(point(2,-2));
    ps.push_back(point(0,-2));


//===============
    // ps.push_back(point(-1,1));
    // ps.push_back(point(1,1));
    // ps.push_back(point(1,-1));
    // ps.push_back(point(-1,-1));

    // ps.push_back(point(0,2));
    // ps.push_back(point(0.5,0.5));
    // ps.push_back(point(2,0));
    // ps.push_back(point(0.5,-0.5));
    // ps.push_back(point(0,-2));
    // ps.push_back(point(-0.5,-0.5));
    // ps.push_back(point(-2,0));
    // ps.push_back(point(-0.5,0.5));
    
    
    // pb.push_back(point(0,2));
    // pb.push_back(point(0.5,0.5));
    // pb.push_back(point(2,0));
    // pb.push_back(point(0.5,-0.5));
    // pb.push_back(point(0,-2));
    // pb.push_back(point(-0.5,-0.5));
    // pb.push_back(point(-2,0));
    // pb.push_back(point(-0.5,0.5));
//=================

    
    // pb.push_back(point(0,2));
    // pb.push_back(point(2,-2));
    // pb.push_back(point(-2,-2));

    pb.push_back(point(1,2));
    pb.push_back(point(3,2));
    pb.push_back(point(3,-2));
    pb.push_back(point(1,-2));
    
    
    // clock_wise(ps);
    // for(int i = 0;i < ps.size();i++){
    //     printf("%f %f cos:%f\n",ps[i].x,ps[i].y,cos(ps[i]-start,point(0,1)));
    // }
    printf("area %f %f %f\n",NPIA(ps,pb),polygon_area(ps),polygon_area(pb));
    // printf("ps %f\n",polygon_area(ps));
    return 0;
}