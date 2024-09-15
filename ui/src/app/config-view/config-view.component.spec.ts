import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ConfigViewComponent } from './config-view.component';

describe('ConfigViewComponent', () => {
  let component: ConfigViewComponent;
  let fixture: ComponentFixture<ConfigViewComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ConfigViewComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ConfigViewComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
